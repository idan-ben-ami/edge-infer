#!/usr/bin/env python3
"""
edge-infer: ONNX -> Rust no_std code generator for microcontrollers.

Reads an ONNX model and generates a complete Rust no_std crate with
inference code. The model becomes code, not data.

Usage:
    python edge_infer.py mnist_cnn.onnx --output ./mnist_model/
"""

import argparse
import os
import re
import sys
import textwrap
from typing import Optional

import numpy as np
import onnx
from onnx import numpy_helper


# ---------------------------------------------------------------------------
# 1. ONNX Parser
# ---------------------------------------------------------------------------

def load_model(path: str):
    """Load and validate an ONNX model, returning ops, weights, and input info."""
    model = onnx.load(path)
    onnx.checker.check_model(model)
    graph = model.graph

    # Extract weights (initializers) as numpy arrays
    weights: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        weights[init.name] = numpy_helper.to_array(init)

    # Extract ops in topological order with parsed attributes
    ops = []
    for node in graph.node:
        attrs = {}
        for a in node.attribute:
            if a.type == onnx.AttributeProto.INTS:
                attrs[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.INT:
                attrs[a.name] = a.i
            elif a.type == onnx.AttributeProto.FLOAT:
                attrs[a.name] = a.f
            elif a.type == onnx.AttributeProto.FLOATS:
                attrs[a.name] = list(a.floats)
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode("utf-8")
        ops.append({
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attrs": attrs,
            "name": node.name,
        })

    # Get input shape from graph inputs (skip initializers)
    init_names = {init.name for init in graph.initializer}
    input_info = None
    for inp in graph.input:
        if inp.name not in init_names:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else 1)
            input_info = {"name": inp.name, "shape": dims}
            break

    if input_info is None:
        print("Error: could not find model input", file=sys.stderr)
        sys.exit(1)

    return ops, weights, input_info


# ---------------------------------------------------------------------------
# 2. Shape Tracker -- walk the graph and compute shapes at each stage
# ---------------------------------------------------------------------------

class ShapeTracker:
    """Tracks tensor shapes through the ONNX graph."""

    def __init__(self, input_name: str, input_shape: list[int],
                 weights: dict[str, np.ndarray]):
        self.shapes: dict[str, list[int]] = {input_name: input_shape}
        self.weights = weights
        for name, arr in weights.items():
            self.shapes[name] = list(arr.shape)

    def get(self, name: str) -> list[int]:
        return self.shapes[name]

    def set(self, name: str, shape: list[int]):
        self.shapes[name] = shape

    def compute_conv_output(self, input_shape: list[int],
                            weight_shape: list[int],
                            pads: list[int],
                            strides: list[int]) -> list[int]:
        """Compute Conv2d output shape.
        input: [N,C,H,W] or [C,H,W], weight: [CO,CI,KH,KW]."""
        if len(input_shape) == 4:
            _, _, h, w = input_shape
        else:
            _, h, w = input_shape
        co, _, kh, kw = weight_shape
        pad_top = pads[0]
        pad_left = pads[1]
        pad_bottom = pads[2] if len(pads) > 2 else pads[0]
        pad_right = pads[3] if len(pads) > 3 else pads[1]
        sh, sw = strides
        oh = (h + pad_top + pad_bottom - kh) // sh + 1
        ow = (w + pad_left + pad_right - kw) // sw + 1
        return [co, oh, ow]

    def compute_pool_output(self, input_shape: list[int],
                            kernel: list[int],
                            strides: list[int]) -> list[int]:
        """Compute MaxPool2d output shape."""
        if len(input_shape) == 4:
            _, c, h, w = input_shape
        else:
            c, h, w = input_shape
        kh, kw = kernel
        sh, sw = strides
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        return [c, oh, ow]


# ---------------------------------------------------------------------------
# 2b. INT8 Symmetric Quantization
# ---------------------------------------------------------------------------

def quantize_tensor_symmetric(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization: zero_point = 0, range [-127, 127].

    Returns (quantized_int8_array, scale) where:
        original ~= quantized * scale
    """
    abs_max = np.max(np.abs(arr))
    scale = float(abs_max / 127.0) if abs_max > 0 else 1.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized, scale


# ---------------------------------------------------------------------------
# 3. Weight Name Sanitizer
# ---------------------------------------------------------------------------

_weight_name_map: dict[str, str] = {}


def reset_name_map():
    global _weight_name_map
    _weight_name_map = {}


def sanitize_weight_name(onnx_name: str) -> str:
    """Convert an ONNX tensor name to a clean Rust constant name.

    Handles PyTorch export patterns like "conv1.weight", "fc1.bias",
    as well as numbered patterns like "onnx::Conv_42".
    """
    if onnx_name in _weight_name_map:
        return _weight_name_map[onnx_name]

    name = onnx_name
    # Replace dots, slashes, double-colons with underscores
    name = re.sub(r"[./:]", "_", name)
    # Remove any remaining invalid chars
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    name_lower = name.lower()

    # Detect layer type
    layer_type = None
    if "conv" in name_lower:
        layer_type = "conv"
    elif any(kw in name_lower for kw in ("fc", "linear", "classifier")):
        layer_type = "fc"

    is_bias = "bias" in name_lower

    if layer_type:
        nums = re.findall(r"\d+", name)
        layer_num = nums[0] if nums else ""
        suffix = "BIAS" if is_bias else "WEIGHT"
        clean = f"{layer_type.upper()}{layer_num}_{suffix}"
    else:
        clean = name.upper()
        if not clean[0].isalpha():
            clean = "W_" + clean

    # Detect collisions — append _2, _3, etc. if name already used
    existing_names = set(_weight_name_map.values())
    final = clean
    counter = 2
    while final in existing_names:
        final = f"{clean}_{counter}"
        counter += 1

    _weight_name_map[onnx_name] = final
    return final


# ---------------------------------------------------------------------------
# 4. Rust Code Emitters
# ---------------------------------------------------------------------------

def format_f32(val: float) -> str:
    """Format a float for Rust with enough precision and explicit type."""
    return f"{val:.8e}_f32"


def format_i8(val: int) -> str:
    """Format an int8 value for Rust."""
    return str(int(val))


def emit_ndarray(arr: np.ndarray, indent: int = 0,
                 is_int: bool = False) -> str:
    """Recursively emit a numpy array as a Rust array literal."""
    prefix = " " * indent
    fmt = format_i8 if is_int else (lambda v: format_f32(float(v)))
    if arr.ndim == 1:
        vals = ", ".join(fmt(v) for v in arr)
        return f"[{vals}]"
    else:
        inner_parts = []
        for i in range(arr.shape[0]):
            inner_parts.append(emit_ndarray(arr[i], indent + 4,
                                            is_int=is_int))
        sep = f",\n{prefix}    "
        return f"[\n{prefix}    {sep.join(inner_parts)},\n{prefix}]"


def rust_type_for_shape(shape: tuple[int, ...],
                        elem_type: str = "f32") -> str:
    """Generate Rust nested array type for a given shape."""
    if len(shape) == 1:
        return f"[{elem_type}; {shape[0]}]"
    else:
        inner = rust_type_for_shape(shape[1:], elem_type)
        return f"[{inner}; {shape[0]}]"


def emit_weights_rs(weights: dict[str, np.ndarray],
                    used_weights: set[str],
                    quantize: str = "none",
                    bias_names: Optional[set[str]] = None) -> str:
    """Generate weights.rs with const arrays for all weights used.

    When quantize='int8', weight tensors (non-bias) are emitted as i8 with
    a companion _SCALE: f32 constant. Biases remain f32.
    """
    lines = [
        "// Generated by edge-infer",
        "#![allow(clippy::excessive_precision)]",
        "#![allow(clippy::unreadable_literal)]",
        "",
    ]
    if bias_names is None:
        bias_names = set()

    # Sort for deterministic output
    for onnx_name in sorted(used_weights):
        arr = weights[onnx_name]
        rust_name = sanitize_weight_name(onnx_name)

        is_bias = onnx_name in bias_names
        should_quantize = quantize == "int8" and not is_bias

        if should_quantize:
            q_arr, scale = quantize_tensor_symmetric(arr)
            rust_type = rust_type_for_shape(q_arr.shape, elem_type="i8")
            val = emit_ndarray(q_arr, 0, is_int=True)
            lines.append(f"pub const {rust_name}: {rust_type} = {val};")
            lines.append(
                f"pub const {rust_name}_SCALE: f32 = {format_f32(scale)};"
            )
        else:
            rust_type = rust_type_for_shape(arr.shape)
            val = emit_ndarray(arr, 0)
            lines.append(f"pub const {rust_name}: {rust_type} = {val};")

        lines.append("")

    return "\n".join(lines)


def emit_ops_rs(quantize: str = "none") -> str:
    """Generate ops.rs with generic operator implementations.

    These are standalone functions using Rust const generics. They work in
    no_std environments with zero allocations -- all buffers are caller-provided.

    When quantize='int8', also emits conv2d_q8 and dense_q8 variants that
    accept i8 weights with a scale factor.
    """
    code = textwrap.dedent("""\
        // Generated by edge-infer
        //
        // Generic inference operators for no_std environments.
        // All buffer sizes are compile-time constants via const generics.

        pub fn conv2d<
            const CI: usize,
            const CO: usize,
            const K: usize,
            const H: usize,
            const W: usize,
            const OH: usize,
            const OW: usize,
            const PAD: usize,
        >(
            input: &[[[f32; W]; H]; CI],
            kernel: &[[[[f32; K]; K]; CI]; CO],
            bias: &[f32; CO],
            output: &mut [[[f32; OW]; OH]; CO],
        ) {
            for co in 0..CO {
                for oh in 0..OH {
                    for ow in 0..OW {
                        let mut sum = bias[co];
                        for ci in 0..CI {
                            for kh in 0..K {
                                for kw in 0..K {
                                    let ih = oh + kh;
                                    let iw = ow + kw;
                                    if ih >= PAD && ih < H + PAD && iw >= PAD && iw < W + PAD {
                                        sum += input[ci][ih - PAD][iw - PAD] * kernel[co][ci][kh][kw];
                                    }
                                }
                            }
                        }
                        output[co][oh][ow] = sum;
                    }
                }
            }
        }

        pub fn relu_inplace(data: &mut [f32]) {
            for x in data.iter_mut() {
                if *x < 0.0 {
                    *x = 0.0;
                }
            }
        }

        pub fn max_pool2d<
            const C: usize,
            const H: usize,
            const W: usize,
            const OH: usize,
            const OW: usize,
        >(
            input: &[[[f32; W]; H]; C],
            output: &mut [[[f32; OW]; OH]; C],
        ) {
            for c in 0..C {
                for oh in 0..OH {
                    for ow in 0..OW {
                        let mut max = f32::NEG_INFINITY;
                        for kh in 0..2 {
                            for kw in 0..2 {
                                let val = input[c][oh * 2 + kh][ow * 2 + kw];
                                if val > max {
                                    max = val;
                                }
                            }
                        }
                        output[c][oh][ow] = max;
                    }
                }
            }
        }

        pub fn dense<const N: usize, const M: usize>(
            input: &[f32; N],
            weights: &[[f32; N]; M],
            bias: &[f32; M],
            output: &mut [f32; M],
        ) {
            for m in 0..M {
                let mut sum = bias[m];
                for n in 0..N {
                    sum += input[n] * weights[m][n];
                }
                output[m] = sum;
            }
        }
    """)

    if quantize == "int8":
        code += textwrap.dedent("""\

        pub fn conv2d_q8<
            const CI: usize,
            const CO: usize,
            const K: usize,
            const H: usize,
            const W: usize,
            const OH: usize,
            const OW: usize,
            const PAD: usize,
        >(
            input: &[[[f32; W]; H]; CI],
            kernel: &[[[[i8; K]; K]; CI]; CO],
            kernel_scale: f32,
            bias: &[f32; CO],
            output: &mut [[[f32; OW]; OH]; CO],
        ) {
            for co in 0..CO {
                for oh in 0..OH {
                    for ow in 0..OW {
                        let mut sum = bias[co];
                        for ci in 0..CI {
                            for kh in 0..K {
                                for kw in 0..K {
                                    let ih = oh + kh;
                                    let iw = ow + kw;
                                    if ih >= PAD && ih < H + PAD && iw >= PAD && iw < W + PAD {
                                        let w = kernel[co][ci][kh][kw] as f32 * kernel_scale;
                                        sum += input[ci][ih - PAD][iw - PAD] * w;
                                    }
                                }
                            }
                        }
                        output[co][oh][ow] = sum;
                    }
                }
            }
        }

        pub fn dense_q8<const N: usize, const M: usize>(
            input: &[f32; N],
            weights: &[[i8; N]; M],
            weight_scale: f32,
            bias: &[f32; M],
            output: &mut [f32; M],
        ) {
            for m in 0..M {
                let mut sum = bias[m];
                for n in 0..N {
                    sum += input[n] * (weights[m][n] as f32 * weight_scale);
                }
                output[m] = sum;
            }
        }
        """)

    return code


def emit_cargo_toml(crate_name: str = "mnist-model") -> str:
    """Generate Cargo.toml for the no_std crate."""
    lib_name = crate_name.replace("-", "_")
    return textwrap.dedent(f"""\
        [package]
        name = "{crate_name}"
        version = "0.1.0"
        edition = "2021"

        [lib]
        name = "{lib_name}"

        [[bin]]
        name = "demo"
        path = "src/bin/demo.rs"
        required-features = ["demo"]

        [features]
        demo = ["dep:cortex-m", "dep:cortex-m-rt", "dep:cortex-m-semihosting", "dep:panic-semihosting"]

        [dependencies]
        cortex-m = {{ version = "0.7", optional = true }}
        cortex-m-rt = {{ version = "0.7", optional = true }}
        cortex-m-semihosting = {{ version = "0.5", optional = true }}
        panic-semihosting = {{ version = "0.6", optional = true }}

        [profile.release]
        opt-level = "z"
        lto = true
        codegen-units = 1
    """)


def emit_demo_rs(input_shape: list[int], lib_name: str) -> str:
    """Generate the QEMU semihosting demo binary."""
    if len(input_shape) == 4:
        c, h, w = input_shape[1], input_shape[2], input_shape[3]
    else:
        c, h, w = input_shape
    return textwrap.dedent(f"""\
        #![no_std]
        #![no_main]

        use cortex_m_rt::entry;
        use cortex_m_semihosting::{{
            debug::{{self, EXIT_SUCCESS}},
            hprintln,
        }};
        use panic_semihosting as _;

        use {lib_name}::predict;

        // Zero input -- replace with real test data for meaningful results
        static INPUT: [[[f32; {w}]; {h}]; {c}] = [[[0.0; {w}]; {h}]; {c}];

        #[entry]
        fn main() -> ! {{
            hprintln!("Running inference on Cortex-M4...");

            let output = predict(&INPUT);
            let predicted = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            hprintln!("Predicted class: {{}}", predicted);

            hprintln!("Logits:");
            for (i, val) in output.iter().enumerate() {{
                hprintln!("  [{{}}] {{:.4}}", i, val);
            }}

            debug::exit(EXIT_SUCCESS);
            loop {{}}
        }}
    """)


def emit_memory_x() -> str:
    """Generate memory.x linker script for QEMU LM3S6965."""
    return textwrap.dedent("""\
        /* LM3S6965 (Stellaris) -- QEMU lm3s6965evb */
        MEMORY
        {
            FLASH : ORIGIN = 0x00000000, LENGTH = 256K
            RAM   : ORIGIN = 0x20000000, LENGTH = 64K
        }
    """)


def emit_cargo_config() -> str:
    """Generate .cargo/config.toml for Cortex-M linking."""
    return textwrap.dedent("""\
        [target.thumbv7em-none-eabihf]
        rustflags = ["-C", "link-arg=-Tlink.x"]
    """)


def crate_name_to_lib(name: str) -> str:
    return name.replace("-", "_")


# ---------------------------------------------------------------------------
# 5. Predict function emitter -- walks the ONNX graph
# ---------------------------------------------------------------------------

class PredictEmitter:
    """Walks the ONNX graph and emits the body of the Rust predict() function.

    Maintains a mapping from ONNX tensor names to Rust variable names and
    tracks whether each tensor is a flat [f32; N] or a 3D [[[f32; W]; H]; C].
    """

    def __init__(self, ops: list[dict], weights: dict[str, np.ndarray],
                 tracker: ShapeTracker, input_name: str,
                 quantize: str = "none"):
        self.ops = ops
        self.weights = weights
        self.tracker = tracker
        self.input_name = input_name
        self.quantize = quantize
        self.used_weights: set[str] = set()
        self.bias_names: set[str] = set()
        self.lines: list[str] = []
        self.prefix_counts: dict[str, int] = {}
        self.tensor_vars: dict[str, str] = {input_name: "input"}
        self.tensor_is_flat: dict[str, bool] = {input_name: False}

    def _new_buf(self, prefix: str = "buf") -> str:
        # Track how many of each prefix we've seen for naming like conv1, conv2
        count = self.prefix_counts.get(prefix, 0) + 1
        self.prefix_counts[prefix] = count
        return f"{prefix}{count}_out"

    def _get_var(self, name: str) -> str:
        return self.tensor_vars.get(name, name)

    def emit(self) -> tuple[str, set[str], set[str]]:
        """Walk all ops and return (predict_body_code, used_weight_names, bias_names)."""
        for op in self.ops:
            handler = {
                "Conv": self._emit_conv,
                "Relu": self._emit_relu,
                "MaxPool": self._emit_maxpool,
                "Gemm": self._emit_gemm,
                "Reshape": self._emit_reshape,
                "Flatten": self._emit_flatten,
                "Add": self._emit_add,
                "MatMul": self._emit_matmul,
            }.get(op["op_type"])

            if handler:
                handler(op)
            else:
                self.lines.append(
                    f"    // TODO: unsupported op {op['op_type']}"
                )

        return "\n".join(self.lines), self.used_weights, self.bias_names

    # -- Conv2d ---------------------------------------------------------------

    def _emit_conv(self, op: dict):
        input_name = op["inputs"][0]
        weight_name = op["inputs"][1]
        bias_name = op["inputs"][2] if len(op["inputs"]) > 2 else None

        input_shape = self.tracker.get(input_name)
        weight_arr = self.weights[weight_name]
        weight_shape = list(weight_arr.shape)  # [CO, CI, KH, KW]

        pads = op["attrs"].get("pads", [0, 0, 0, 0])
        strides = op["attrs"].get("strides", [1, 1])

        if strides != [1, 1]:
            print(f"WARNING: Conv with stride {strides} is not yet supported "
                  f"(generated code will be incorrect)", file=sys.stderr)

        out_shape = self.tracker.compute_conv_output(
            input_shape, weight_shape, pads, strides
        )
        co, oh, ow = out_shape

        if len(input_shape) == 4:
            ci, h, w = input_shape[1], input_shape[2], input_shape[3]
        else:
            ci, h, w = input_shape

        kh = weight_shape[2]
        pad = pads[0]  # Assume symmetric padding for the spike

        output_name = op["outputs"][0]
        buf = self._new_buf("conv")

        self.used_weights.add(weight_name)
        w_rust = sanitize_weight_name(weight_name)

        if bias_name and bias_name in self.weights:
            self.used_weights.add(bias_name)
            self.bias_names.add(bias_name)
            b_rust = sanitize_weight_name(bias_name)
        else:
            b_rust = None

        input_var = self._get_var(input_name)

        self.lines.append("")
        self.lines.append(
            f"    // Conv: {ci}x{h}x{w} -> {co}x{oh}x{ow}"
            f" (kernel={kh}x{kh}, pad={pad})"
        )
        self.lines.append(
            f"    let mut {buf} = [[[0.0f32; {ow}]; {oh}]; {co}];"
        )

        bias_arg = (
            f"&weights::{b_rust}" if b_rust
            else f"&[0.0f32; {co}]"
        )
        # Ensure we pass a reference
        input_ref = input_var if input_var.startswith("&") or input_var == "input" else f"&{input_var}"

        if self.quantize == "int8":
            self.lines.append(
                f"    ops::conv2d_q8::<{ci}, {co}, {kh}, {h}, {w}, {oh}, {ow}, {pad}>("
            )
            self.lines.append(
                f"        {input_ref}, &weights::{w_rust}, weights::{w_rust}_SCALE, {bias_arg}, &mut {buf},"
            )
        else:
            self.lines.append(
                f"    ops::conv2d::<{ci}, {co}, {kh}, {h}, {w}, {oh}, {ow}, {pad}>("
            )
            self.lines.append(
                f"        {input_ref}, &weights::{w_rust}, {bias_arg}, &mut {buf},"
            )
        self.lines.append(f"    );")

        self.tensor_vars[output_name] = buf
        self.tensor_is_flat[output_name] = False
        self.tracker.set(output_name, out_shape)

    # -- ReLU -----------------------------------------------------------------

    def _emit_relu(self, op: dict):
        input_name = op["inputs"][0]
        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)
        shape = self.tracker.get(input_name)

        flat_size = 1
        for d in shape:
            flat_size *= d

        if self.tensor_is_flat.get(input_name, False):
            self.lines.append(f"    // ReLU in-place ({flat_size} elements)")
            self.lines.append(f"    ops::relu_inplace(&mut {input_var});")
        else:
            # 3D buffer -- cast to flat &mut [f32] for in-place relu
            shape_str = "x".join(str(d) for d in shape)
            self.lines.append(f"    // ReLU in-place ({shape_str})")
            self.lines.append(
                f"    // SAFETY: contiguous f32 array, same total size"
            )
            self.lines.append(
                f"    ops::relu_inplace(unsafe {{ "
                f"core::slice::from_raw_parts_mut("
                f"{input_var}.as_mut_ptr() as *mut f32, {flat_size}) }});"
            )

        # Output aliases the same buffer (in-place op)
        self.tensor_vars[output_name] = input_var
        self.tensor_is_flat[output_name] = self.tensor_is_flat.get(
            input_name, False
        )
        self.tracker.set(output_name, shape)

    # -- MaxPool2d ------------------------------------------------------------

    def _emit_maxpool(self, op: dict):
        input_name = op["inputs"][0]
        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)
        input_shape = self.tracker.get(input_name)

        kernel = op["attrs"].get("kernel_shape", [2, 2])
        strides = op["attrs"].get("strides", kernel)

        out_shape = self.tracker.compute_pool_output(
            input_shape, kernel, strides
        )
        c, oh, ow = out_shape

        if len(input_shape) == 4:
            h, w = input_shape[2], input_shape[3]
        else:
            h, w = input_shape[1], input_shape[2]

        buf = self._new_buf("pool")

        self.lines.append("")
        self.lines.append(f"    // MaxPool: {c}x{h}x{w} -> {c}x{oh}x{ow}")
        self.lines.append(
            f"    let mut {buf} = [[[0.0f32; {ow}]; {oh}]; {c}];"
        )
        self.lines.append(
            f"    ops::max_pool2d::<{c}, {h}, {w}, {oh}, {ow}>"
            f"(&{input_var}, &mut {buf});"
        )

        self.tensor_vars[output_name] = buf
        self.tensor_is_flat[output_name] = False
        self.tracker.set(output_name, out_shape)

    # -- Reshape / Flatten ----------------------------------------------------

    def _emit_reshape(self, op: dict):
        input_name = op["inputs"][0]
        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)
        input_shape = self.tracker.get(input_name)

        flat_size = 1
        for d in input_shape:
            flat_size *= d

        shape_str = "x".join(str(d) for d in input_shape)

        self.lines.append("")
        self.lines.append(
            f"    // Reshape: {shape_str} -> flat [{flat_size}]"
        )
        self.lines.append(
            f"    // SAFETY: [[[f32; {input_shape[-1]}]; {input_shape[-2]}]; {input_shape[-3]}] "
            f"is layout-identical to [f32; {flat_size}]"
        )
        self.lines.append(
            f"    let flat: &[f32; {flat_size}] = unsafe {{ "
            f"core::mem::transmute(&{input_var}) }};"
        )

        self.tensor_vars[output_name] = f"*flat"
        self.tensor_is_flat[output_name] = True
        self.tracker.set(output_name, [flat_size])

    def _emit_flatten(self, op: dict):
        """Handle Flatten op (semantically identical to reshape for our use)."""
        input_name = op["inputs"][0]
        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)
        input_shape = self.tracker.get(input_name)

        axis = op["attrs"].get("axis", 1)
        # Flatten from axis onward. For typical CNN with batch dim stripped,
        # axis=1 means flatten all spatial dims.
        start = axis if len(input_shape) == 4 else max(0, axis - 1)
        flat_size = 1
        for d in input_shape[start:]:
            flat_size *= d

        shape_str = "x".join(str(d) for d in input_shape)

        self.lines.append("")
        self.lines.append(
            f"    // Flatten: {shape_str} -> flat [{flat_size}]"
        )
        self.lines.append(
            f"    // SAFETY: contiguous f32 arrays, same total element count"
        )
        self.lines.append(
            f"    let flat: &[f32; {flat_size}] = unsafe {{ "
            f"core::mem::transmute(&{input_var}) }};"
        )

        self.tensor_vars[output_name] = "*flat"
        self.tensor_is_flat[output_name] = True
        self.tracker.set(output_name, [flat_size])

    # -- Gemm (Dense / Linear) -----------------------------------------------

    def _emit_gemm(self, op: dict):
        input_name = op["inputs"][0]
        weight_name = op["inputs"][1]
        bias_name = op["inputs"][2] if len(op["inputs"]) > 2 else None

        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)

        weight_arr = self.weights[weight_name]
        trans_b = op["attrs"].get("transB", 0)

        # Gemm: Y = alpha * A * B' + beta * C
        # With transB=1 (typical for PyTorch Linear): W is [out, in]
        # With transB=0: W is [in, out]
        if trans_b:
            out_features, in_features = weight_arr.shape
        else:
            in_features, out_features = weight_arr.shape

        self.used_weights.add(weight_name)
        w_rust = sanitize_weight_name(weight_name)

        if bias_name and bias_name in self.weights:
            self.used_weights.add(bias_name)
            self.bias_names.add(bias_name)
            b_rust = sanitize_weight_name(bias_name)
        else:
            b_rust = None

        buf = self._new_buf("fc")

        # Build the reference to the input
        if input_var.startswith("*"):
            ref_var = f"&{input_var[1:]}"
        else:
            ref_var = f"&{input_var}"

        bias_arg = (
            f"&weights::{b_rust}" if b_rust
            else f"&[0.0f32; {out_features}]"
        )

        self.lines.append("")
        self.lines.append(
            f"    // Dense (Gemm): {in_features} -> {out_features}"
        )
        self.lines.append(f"    let mut {buf} = [0.0f32; {out_features}];")

        if trans_b:
            # Weights are [out, in] -- matches our dense() signature
            if self.quantize == "int8":
                self.lines.append(
                    f"    ops::dense_q8::<{in_features}, {out_features}>"
                    f"({ref_var}, &weights::{w_rust}, weights::{w_rust}_SCALE, {bias_arg}, &mut {buf});"
                )
            else:
                self.lines.append(
                    f"    ops::dense::<{in_features}, {out_features}>"
                    f"({ref_var}, &weights::{w_rust}, {bias_arg}, &mut {buf});"
                )
        else:
            # Weights are [in, out] -- need manual loop with transposed access
            # (quantized transB=0 not supported yet -- uncommon path)
            self.lines.append(
                f"    // Note: transB=0, weights are [{in_features}][{out_features}]"
            )
            b_inline = (
                f"weights::{b_rust}[m]" if b_rust else "0.0"
            )
            if self.quantize == "int8":
                self.lines.append(
                    f"    for m in 0..{out_features} {{"
                )
                self.lines.append(
                    f"        let mut sum = {b_inline};"
                )
                self.lines.append(
                    f"        for n in 0..{in_features} {{"
                )
                self.lines.append(
                    f"            sum += ({ref_var})[n] * (weights::{w_rust}[n][m] as f32 * weights::{w_rust}_SCALE);"
                )
                self.lines.append(f"        }}")
                self.lines.append(f"        {buf}[m] = sum;")
                self.lines.append(f"    }}")
            else:
                self.lines.append(
                    f"    for m in 0..{out_features} {{"
                )
                self.lines.append(
                    f"        let mut sum = {b_inline};"
                )
                self.lines.append(
                    f"        for n in 0..{in_features} {{"
                )
                self.lines.append(
                    f"            sum += ({ref_var})[n] * weights::{w_rust}[n][m];"
                )
                self.lines.append(f"        }}")
                self.lines.append(f"        {buf}[m] = sum;")
                self.lines.append(f"    }}")

        self.tensor_vars[output_name] = buf
        self.tensor_is_flat[output_name] = True
        self.tracker.set(output_name, [out_features])

    # -- Add (standalone bias add) -------------------------------------------

    def _emit_add(self, op: dict):
        """Handle standalone Add -- typically bias after MatMul."""
        input_a = op["inputs"][0]
        input_b = op["inputs"][1]
        output_name = op["outputs"][0]

        # Figure out which input is data and which is a weight (bias)
        if input_b in self.weights:
            data_name, bias_name = input_a, input_b
        elif input_a in self.weights:
            data_name, bias_name = input_b, input_a
        else:
            self.lines.append(
                f"    // TODO: Add with two non-weight tensors"
            )
            self.tensor_vars[output_name] = self._get_var(input_a)
            self.tensor_is_flat[output_name] = self.tensor_is_flat.get(
                input_a, True
            )
            self.tracker.set(output_name, self.tracker.get(input_a))
            return

        data_var = self._get_var(data_name)
        data_shape = self.tracker.get(data_name)
        bias_arr = self.weights[bias_name]

        self.used_weights.add(bias_name)
        self.bias_names.add(bias_name)
        b_rust = sanitize_weight_name(bias_name)

        size = data_shape[-1] if data_shape else bias_arr.shape[0]

        self.lines.append(f"    // Add bias: {size} elements")

        if data_var.startswith("*"):
            # Dereferenced pointer -- need a mutable owned copy
            buf = self._new_buf("biased")
            self.lines.append(f"    let mut {buf} = {data_var};")
            self.lines.append(
                f"    for i in 0..{size} {{ {buf}[i] += weights::{b_rust}[i]; }}"
            )
            self.tensor_vars[output_name] = buf
        else:
            self.lines.append(
                f"    for i in 0..{size} {{ "
                f"{data_var}[i] += weights::{b_rust}[i]; }}"
            )
            self.tensor_vars[output_name] = data_var

        self.tensor_is_flat[output_name] = self.tensor_is_flat.get(
            data_name, True
        )
        self.tracker.set(output_name, data_shape)

    # -- MatMul (standalone, without bias) ------------------------------------

    def _emit_matmul(self, op: dict):
        """Handle standalone MatMul (usually followed by Add for bias)."""
        input_name = op["inputs"][0]
        weight_name = op["inputs"][1]
        output_name = op["outputs"][0]
        input_var = self._get_var(input_name)

        weight_arr = self.weights[weight_name]
        # MatMul: Y = X @ W, where W is [in_features, out_features]
        in_features, out_features = weight_arr.shape

        self.used_weights.add(weight_name)
        w_rust = sanitize_weight_name(weight_name)

        buf = self._new_buf("mm")

        if input_var.startswith("*"):
            ref_var = f"&{input_var[1:]}"
        else:
            ref_var = f"&{input_var}"

        self.lines.append("")
        self.lines.append(
            f"    // MatMul: {in_features} -> {out_features}"
        )
        self.lines.append(f"    let mut {buf} = [0.0f32; {out_features}];")
        self.lines.append(f"    for m in 0..{out_features} {{")
        self.lines.append(f"        let mut sum = 0.0f32;")
        self.lines.append(f"        for n in 0..{in_features} {{")
        self.lines.append(
            f"            sum += ({ref_var})[n] * weights::{w_rust}[n][m];"
        )
        self.lines.append(f"        }}")
        self.lines.append(f"        {buf}[m] = sum;")
        self.lines.append(f"    }}")

        self.tensor_vars[output_name] = buf
        self.tensor_is_flat[output_name] = True
        self.tracker.set(output_name, [out_features])


# ---------------------------------------------------------------------------
# 6. lib.rs emitter
# ---------------------------------------------------------------------------

def emit_lib_rs(ops: list[dict], weights: dict[str, np.ndarray],
                input_info: dict,
                quantize: str = "none") -> tuple[str, set[str], set[str]]:
    """Generate lib.rs with the predict() function.

    Returns (lib_rs_content, set_of_used_weight_names, set_of_bias_names).
    """
    input_name = input_info["name"]
    input_shape = input_info["shape"]

    # Remove batch dimension if present: [1,1,28,28] -> [1,28,28]
    if len(input_shape) == 4:
        display_shape = input_shape[1:]
    else:
        display_shape = input_shape

    tracker = ShapeTracker(input_name, input_shape, weights)
    emitter = PredictEmitter(ops, weights, tracker, input_name,
                             quantize=quantize)
    body, used_weights, bias_names = emitter.emit()

    # Determine the final output tensor
    last_op = ops[-1]
    final_output = last_op["outputs"][0]
    final_shape = tracker.get(final_output)
    final_var = emitter.tensor_vars.get(final_output, "output")
    final_size = 1
    for d in final_shape:
        final_size *= d

    c, h, w = display_shape
    input_type = f"&[[[f32; {w}]; {h}]; {c}]"

    code = textwrap.dedent(f"""\
        // Generated by edge-infer
        #![no_std]

        #[allow(dead_code)]
        mod ops;
        mod weights;

        /// Run inference on a single input image.
        ///
        /// Input shape: [{c}][{h}][{w}] (channels, height, width)
        /// Output: [{final_size}] logits
        pub fn predict(input: {input_type}) -> [f32; {final_size}] {{
    """)
    code += body
    code += f"\n\n    {final_var}\n}}\n"

    return code, used_weights, bias_names


# ---------------------------------------------------------------------------
# 7. File writer
# ---------------------------------------------------------------------------

def write_crate(output_dir: str, ops: list[dict],
                weights: dict[str, np.ndarray], input_info: dict,
                quantize: str = "none"):
    """Write the complete Rust crate to output_dir."""
    reset_name_map()

    src_dir = os.path.join(output_dir, "src")
    bin_dir = os.path.join(src_dir, "bin")
    cargo_dir = os.path.join(output_dir, ".cargo")
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(cargo_dir, exist_ok=True)

    # Derive crate name from output directory
    crate_name = re.sub(r"[^a-zA-Z0-9_-]", "-",
                        os.path.basename(os.path.normpath(output_dir)))
    if not crate_name or crate_name == ".":
        crate_name = "generated-model"
    lib_name = crate_name.replace("-", "_")

    # Generate lib.rs first to discover which weights are actually used
    lib_rs, used_weights, bias_names = emit_lib_rs(
        ops, weights, input_info, quantize=quantize
    )

    cargo_toml = emit_cargo_toml(crate_name=crate_name)
    ops_rs = emit_ops_rs(quantize=quantize)
    weights_rs = emit_weights_rs(weights, used_weights,
                                 quantize=quantize,
                                 bias_names=bias_names)
    demo_rs = emit_demo_rs(input_info["shape"], lib_name=lib_name)
    memory_x = emit_memory_x()
    cargo_config = emit_cargo_config()

    files = {
        "Cargo.toml": cargo_toml,
        "src/lib.rs": lib_rs,
        "src/ops.rs": ops_rs,
        "src/weights.rs": weights_rs,
        "src/bin/demo.rs": demo_rs,
        "memory.x": memory_x,
        ".cargo/config.toml": cargo_config,
    }

    for rel_path, content in files.items():
        full_path = os.path.join(output_dir, rel_path)
        # Don't overwrite Cargo.toml if it exists (user may have added deps)
        if rel_path == "Cargo.toml" and os.path.exists(full_path):
            print(f"  skipped {full_path} (already exists)")
            continue
        with open(full_path, "w") as f:
            f.write(content)
        print(f"  wrote {full_path}")

    # Summary
    total_params = sum(
        arr.size for name, arr in weights.items() if name in used_weights
    )
    if quantize == "int8":
        # Weights are i8 (1 byte each), biases are f32 (4 bytes each)
        bias_params = sum(
            weights[n].size for n in used_weights if n in bias_names
        )
        weight_params = total_params - bias_params
        weight_bytes = weight_params * 1 + bias_params * 4
        # Also account for one f32 scale per quantized tensor
        num_quantized = len(used_weights - bias_names)
        weight_bytes += num_quantized * 4
        quant_note = " (INT8 weights + f32 biases/scales)"
    else:
        weight_bytes = total_params * 4  # f32 = 4 bytes
        quant_note = ""
    print(f"\nSummary:")
    print(f"  Quantization: {quantize}")
    print(f"  Operators:  {len(ops)}")
    print(
        f"  Weights:    {len(used_weights)} tensors, "
        f"{total_params:,} parameters ({weight_bytes:,} bytes{quant_note})"
    )
    print(f"  Output dir: {output_dir}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "edge-infer: ONNX to Rust no_std code generator "
            "for microcontrollers"
        )
    )
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument(
        "--output", "-o",
        default="./generated/",
        help="Output directory for generated Rust crate (default: ./generated/)",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "int8"],
        default="none",
        help="Weight quantization (default: none = f32)",
    )
    args = parser.parse_args()

    print(f"edge-infer: {args.model} -> {args.output}")
    if args.quantize != "none":
        print(f"  quantization: {args.quantize}")
    print()

    # Load and parse
    print("Loading ONNX model...")
    ops, weights, input_info = load_model(args.model)

    print(f"  Input: {input_info['name']} shape={input_info['shape']}")
    print(f"  Ops:   {len(ops)} nodes")
    print(f"  Weights: {len(weights)} tensors")
    print()

    # Print op summary
    print("Graph:")
    for i, op in enumerate(ops):
        detail = ""
        if op["op_type"] == "Conv":
            ks = op["attrs"].get("kernel_shape", "?")
            pads = op["attrs"].get("pads", "?")
            detail = f" kernel={ks} pads={pads}"
        elif op["op_type"] == "MaxPool":
            ks = op["attrs"].get("kernel_shape", "?")
            detail = f" kernel={ks}"
        elif op["op_type"] == "Gemm":
            tb = op["attrs"].get("transB", 0)
            detail = f" transB={tb}"
        print(f"  [{i}] {op['op_type']}{detail}")
    print()

    # Generate
    print("Generating Rust crate...")
    write_crate(args.output, ops, weights, input_info,
                quantize=args.quantize)
    print(f"\nDone! Try: cd {args.output} && cargo build --release")


if __name__ == "__main__":
    main()
