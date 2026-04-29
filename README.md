# edge-infer

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg?style=flat-square)](#license)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![no_std](https://img.shields.io/badge/no__std-yes-success.svg?style=flat-square)](https://docs.rust-embedded.org/book/intro/no-std.html)
[![Cortex-M tested](https://img.shields.io/badge/cortex--m4-tested-success.svg?style=flat-square)](#qemu-demo)

**ONNX models in, Rust `no_std` code out. Your model becomes code, not data.**

edge-infer reads an ONNX neural network and generates a complete Rust `no_std` crate --
pure inference code with weights baked in as constants. No runtime, no interpreter,
no heap allocation. The generated code compiles directly for bare-metal microcontrollers.

```
                edge-infer
ONNX model ──────────────────> Rust no_std crate ────> Cortex-M / RISC-V / any MCU
 (model.onnx)   (code gen)     (predict function)      (bare metal, zero alloc)
```

## Why this exists

Open-source TinyML isn't dead — Apache TVM / microTVM, LiteRT, uTensor,
ExecuTorch, [onnx2c](https://github.com/kraiskil/onnx2c), and
[MicroFlow](https://github.com/matteocarnelos/microflow-rs) all exist.
What got thinner: the *vendor-neutral, end-to-end, friction-free* corner.
[Edge Impulse was acquired by Qualcomm](https://www.edgeimpulse.com/blog/edge-impulse-qualcomm-acquisition/)
in March 2025 and tilted toward Dragonwing hardware. Nobody has
productized vendor-neutral, end-to-end, ONNX → Rust `no_std` + INT8.

[MicroFlow](https://github.com/matteocarnelos/microflow-rs) (University
of Padova thesis) proved Rust can match or beat TFLite Micro on
bare-metal MCUs by embedding the model at compile time;
[onnx2c](https://github.com/kraiskil/onnx2c) showed the same idea works
for ONNX → C. edge-infer is the productized version of that approach
for Rust.

The trick is simple: eliminate the interpreter, eliminate the runtime overhead.
TFLite Micro ships a ~60-100KB C++ runtime that parses FlatBuffers, dispatches ops,
and manages memory at runtime. edge-infer generates a single `predict()` function
with all shapes known at compile time, all buffers on the stack, and all weights
as `const` arrays. The compiler sees the full picture and optimizes accordingly.

## Quickstart

```bash
# Install dependencies (requires Python 3.11+ and uv)
uv sync

# Generate a Rust crate from the included MNIST model
uv run python edge_infer.py examples/mnist/mnist_cnn.onnx --output ./mnist_model/ --quantize int8
```

The generated crate is self-contained -- no runtime dependencies:

```bash
cd mnist_model
cargo build --release                                        # build for host
cargo build --target thumbv7em-none-eabihf --release         # cross-compile for Cortex-M4
```

Six examples ship with the repo across three distinct architectures
(CNN, pure-MLP, MLP-on-engineered-features), six dataset / RAM-budget
combinations: two MNIST CNNs (digits + clothing) on the same topology,
two MLPs (Iris's 4→16→8→3 and the CWRU bearing-fault 32→64→32→2),
and two CIFAR-10 CNN variants sized for the lm3s6965evb 64 KB and
mps2-an386 4 MB SRAM ceilings. They cover image classification, tabular
classification, and fault detection. All run end-to-end through the
same code generator.

| Example | Topology | Domain | Test accuracy | INT8 weights | Peak stack (measured) | Target board |
|---|---|---|---|---|---|---|
| [`examples/mnist/`](examples/mnist/) | CNN | Handwritten digits | 98.85% (INT8) | 52 KB | 47 KB | lm3s6965evb (64 KB) |
| [`examples/fashion_mnist/`](examples/fashion_mnist/) | CNN | Clothing categories | 89.04% | 52 KB | 47 KB | lm3s6965evb |
| [`examples/iris/`](examples/iris/) | MLP 4→16→8→3 | Tabular classifier | 96.67% | 336 B | ≤ 256 B | lm3s6965evb |
| [`examples/vibration_anomaly/`](examples/vibration_anomaly/) | MLP 32→64→32→2 | **Real CWRU bearing-fault data** | 100% on a per-recording fair split (held-out fault recordings + temporal-gap healthy split — see [example README](examples/vibration_anomaly/README.md)) | 4.5 KB | 416 B | lm3s6965evb |
| [`examples/cifar10_tiny/`](examples/cifar10_tiny/) | small CNN | Image classifier (limit demo) | 54.56% | 10 KB | 20 KB | lm3s6965evb |
| [`examples/cifar10_mps2/`](examples/cifar10_mps2/) | bigger CNN | Image classifier (production-shape) | 63.96% | 39 KB | 72 KB | mps2-an386 (4 MB SRAM) |

All accuracies above are the result of running each `examples/*/train.py`
fresh on a CPU with `SEED = 42` pinned across `random`, `numpy`,
`torch`. Reproducible byte-identical: `uv run --extra train python
examples/<name>/train.py` will print the same number on the same
machine + toolchain.

"Peak stack (measured)" is the real high-water mark from running
each example's generated `stack_probe` binary on QEMU (which emulates
RAM correctly, so the number matches silicon). Reproduce with
`cargo run --bin stack_probe --features demo --release`. Methodology
and per-example breakdown in [BENCHMARKS.md](BENCHMARKS.md#peak-stack-usage-real-measured).
Heap usage is zero across the board — all buffers are stack-allocated.

A note on the MNIST 47 KB number: an earlier draft of this table said
"~28 KB" computed as the largest *single-layer* live activation pair.
That was wrong — Rust doesn't reuse stack slots across `let mut`
declarations even with LTO, so the real footprint is the **sum** of
all activation buffers. Optimizing the codegen to reuse slots across
non-overlapping buffer lifetimes is on the post-launch roadmap; until
then, treat 47 KB as the honest number for this topology. It's why
cifar10_mps2 doesn't fit on the 64 KB lm3s6965evb (72 KB needed) and
needs the 4 MB SRAM mps2-an386 board.

All generated `predict()` functions are browsable on GitHub without compiling —
e.g. [`examples/vibration_anomaly/generated/src/lib.rs`](examples/vibration_anomaly/generated/src/lib.rs)
or [`examples/iris/generated/src/lib.rs`](examples/iris/generated/src/lib.rs).

The five lm3s6965evb examples all fit in 64 KB SRAM. CIFAR-10's full-size
variant doesn't — that's why two CIFAR-10 examples ship: `cifar10_tiny`
(shrunk to fit the smallest demo board) and `cifar10_mps2` (full-size on a
board sized for real ML deployments).

To retrain any example yourself (optional, downloads ~2GB of PyTorch):

```bash
uv sync --extra train
uv run python examples/mnist/train.py            # any of the 6
```

## Generated code

This is the actual `predict()` function generated from a 2-layer MNIST CNN
with INT8 quantization. Not pseudocode -- this is what `edge_infer.py --quantize int8`
produces (also browsable at [`examples/mnist/generated/src/lib.rs`](examples/mnist/generated/src/lib.rs)):

```rust
// Generated by edge-infer
#![no_std]

mod ops;
mod weights;

/// Run inference on a single input image.
///
/// Input shape: [1][28][28] (channels, height, width)
/// Output: [10] logits
pub fn predict(input: &[[[f32; 28]; 28]; 1]) -> [f32; 10] {

    // Conv: 1x28x28 -> 8x28x28 (kernel=3x3, pad=1)
    let mut conv1_out = [[[0.0f32; 28]; 28]; 8];
    ops::conv2d_q8::<1, 8, 3, 28, 28, 28, 28, 1>(
        input, &weights::CONV1_WEIGHT, weights::CONV1_WEIGHT_SCALE, &weights::CONV1_BIAS, &mut conv1_out,
    );
    // ReLU in-place (8x28x28)
    ops::relu_inplace(conv1_out.as_mut_slice().as_flattened_mut().as_flattened_mut());

    // MaxPool: 8x28x28 -> 8x14x14
    let mut pool1_out = [[[0.0f32; 14]; 14]; 8];
    ops::max_pool2d::<8, 28, 28, 14, 14>(&conv1_out, &mut pool1_out);

    // Conv: 8x14x14 -> 16x14x14 (kernel=3x3, pad=1)
    let mut conv2_out = [[[0.0f32; 14]; 14]; 16];
    ops::conv2d_q8::<8, 16, 3, 14, 14, 14, 14, 1>(
        &pool1_out, &weights::CONV2_WEIGHT, weights::CONV2_WEIGHT_SCALE, &weights::CONV2_BIAS, &mut conv2_out,
    );
    // ReLU in-place (16x14x14)
    ops::relu_inplace(conv2_out.as_mut_slice().as_flattened_mut().as_flattened_mut());

    // MaxPool: 16x14x14 -> 16x7x7
    let mut pool2_out = [[[0.0f32; 7]; 7]; 16];
    ops::max_pool2d::<16, 14, 14, 7, 7>(&conv2_out, &mut pool2_out);

    // Reshape: 16x7x7 -> flat [784]
    // SAFETY: Rust arrays are tightly packed with no padding
    // (https://doc.rust-lang.org/reference/type-layout.html#array-layout),
    // so [[[f32; 7]; 7]; 16] is bit-identical to [f32; 784].
    // `<[T]>::as_flattened()` is the safe alternative but returns &[f32]
    // not &[f32; N] — downstream Gemm wants the fixed-size array.
    let flat: &[f32; 784] = unsafe { core::mem::transmute(&pool2_out) };

    // Dense (Gemm): 784 -> 64
    let mut fc1_out = [0.0f32; 64];
    ops::dense_q8::<784, 64>(&flat, &weights::FC1_WEIGHT, weights::FC1_WEIGHT_SCALE, &weights::FC1_BIAS, &mut fc1_out);
    // ReLU in-place (64 elements)
    ops::relu_inplace(&mut fc1_out);

    // Dense (Gemm): 64 -> 10
    let mut fc2_out = [0.0f32; 10];
    ops::dense_q8::<64, 10>(&fc1_out, &weights::FC2_WEIGHT, weights::FC2_WEIGHT_SCALE, &weights::FC2_BIAS, &mut fc2_out);

    fc2_out
}
```

Every intermediate buffer has a compile-time-known size. No `Vec`, no `alloc`,
no heap. The Rust compiler monomorphizes each op call with concrete dimensions,
enabling dead-code elimination and loop unrolling.

## QEMU demo

The generated crate includes a semihosting demo binary for running on
QEMU without real hardware:

```bash
# macOS: brew install qemu | Ubuntu: apt install qemu-system-arm
rustup target add thumbv7em-none-eabihf

cd mnist_model
cargo build --bin demo --features demo --target thumbv7em-none-eabihf --release
qemu-system-arm \
    -cpu cortex-m4 \
    -machine lm3s6965evb \
    -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/demo
```

Output (with a real MNIST test image loaded as input):

```
Running inference on Cortex-M4...
Predicted class: 0
Logits:
  [0] 10.9073
  [1] -8.3618
  [2] -4.6433
  [3] -6.1916
  [4] -6.9860
  [5] -0.2341
  [6] 1.3831
  [7] -1.9843
  [8] -3.8532
  [9] -4.0789
```

Logits match the Python/ONNX reference within 1e-4. The generated demo
uses a zero input by default -- replace `INPUT` with real data for
meaningful predictions.

The QEMU demo binary above is built with `--bin demo --features demo`
which pulls in `cortex-m-semihosting` + `panic-semihosting` (~21 KB
overhead) so the host can read the model's output. For production
firmware, build the `minimal` profile instead:

```bash
cargo build --bin minimal --features minimal --target thumbv7em-none-eabihf --release
arm-none-eabi-size target/thumbv7em-none-eabihf/release/minimal
```

That's the binary the benchmark numbers below are measured against —
the firmware-shippable artifact, with `panic-halt` and no host printing.

## Supported targets

edge-infer's output is plain `no_std` Rust — it runs on whatever board
your toolchain supports. Verified on two QEMU machines:

| Board | RAM | Examples that fit | Notes |
|---|---|---|---|
| `lm3s6965evb` (Stellaris, Cortex-M3) | 64 KB | MNIST, Fashion-MNIST, Iris, vibration anomaly, CIFAR-10 (tiny) | The smallest realistic ML target — what you'd use to prove "yes, it really fits." QEMU's `-cpu cortex-m4` overrides the machine default to host the M4F firmware target. |
| `mps2-an386` (Cortex-M4) | 4 MB code + 4 MB data | All of the above + full-size CIFAR-10 | Project uses SSRAM1 (4 MB FLASH at `0x00000000`) + SSRAM2 (4 MB RAM at `0x20000000`); the board also exposes a 16 MB PSRAM region we don't currently link. Better proxy for an STM32F4/F7/H7 / nRF53 / ESP32-S3-class deployment. |

The full-size CIFAR-10 model (39 K params, **72 KB measured stack**)
exceeds the lm3s6965evb's 64 KB SRAM entirely — measured by running
`stack_probe` on the mps2-an386 (which has the headroom). Two CIFAR-10
examples ship to make the trade-off explicit: shrink to fit
(`cifar10_tiny`, ~10K params, 20 KB stack, 55% accuracy) or move to a
board sized for real ML (`cifar10_mps2`, full model, 64% accuracy on
4 MB SRAM).

> **Why M3 silicon executing M4F firmware in QEMU is fine:** the
> `lm3s6965evb` chip is Cortex-M3, but QEMU's `-cpu cortex-m4` flag
> swaps in an M4F core and runs the `thumbv7em-none-eabihf` instruction
> stream the toolchain emits. The machine model contributes the memory
> map and peripherals; the CPU model contributes the ISA. This is a
> standard QEMU pattern for testing M4 firmware without a board, and
> all the bundled examples build and run under it. For a real M3
> deployment, `thumbv7m-none-eabi` (no FP) is the right target — drop
> a [bug report](https://github.com/idan-ben-ami/edge-infer/issues/new?template=hardware-test-report.yml)
> if you want CI verification of that path.

## Benchmarks

MNIST CNN (2x Conv + 2x Dense, ~51K parameters) on Cortex-M4:

| Approach | Flash | Peak RAM | Runtime overhead | Heap | Accuracy | Notes |
|---|---|---|---|---|---|---|
| **edge-infer (INT8)** | **54 KB** | **47 KB stack (measured)** | **None** | **0** | **98.85%** | This MNIST topology, this toolchain. Apples-to-apples. |
| **edge-infer (f32)** | **204 KB** | **47 KB stack (measured)** | **None** | **0** | **98.82%** | This MNIST topology, this toolchain. Apples-to-apples. |
| TFLite Micro (all-ops, my local build) | 447 KB | Tensor arena (varies) | Runtime dispatch, FlatBuffer parsing | Tensor arena | n/a (lib only) | Same Cortex-M4 toolchain, no op-resolver pruning. **The apples-to-apples baseline.** |
| TFLite Micro (optimized w/ op-resolver, *cited*) | ~105 KB | Tensor arena | Same | Same | n/a (different model) | Published 2021 nRF52840 figure ([arxiv 2112.01319](https://arxiv.org/abs/2112.01319)) — different chip, different MNIST topology, careful op-resolver pruning. **Reference, not apples-to-apples.** |
| TFLite Micro (typical real-world, *cited*) | ~275 KB | Tensor arena | Same | Same | n/a (different model) | Same paper's "real-world" figure. **Reference, not apples-to-apples.** |

The honest comparison: **54 KB vs my own 447 KB all-ops TFLite Micro
build on the same toolchain** — that's an 8× win on flash, no
methodology asterisks. The 105 KB and 275 KB rows are external
reference points (different chip, different MNIST topology), kept in
the table for context but explicitly marked as cited rather than
measured. If you want the cleanest apples-to-apples claim, use the
447 KB row.

edge-infer flash numbers measure the `minimal` binary
(`cargo build --bin minimal --features minimal`), which uses `panic-halt`
and no semihosting — i.e., what you'd actually flash to a real MCU.
Compile flags: `opt-level="z"`, LTO, full release. Binary size measured
with `arm-none-eabi-size` on the `text + data` columns. The `demo`
binary used for QEMU below adds ~21 KB of `cortex-m-semihosting` and
`panic-semihosting` for the host-printable demo output; that's
debugging convenience, not the firmware-shippable size.

A fully apples-to-apples op-resolver-pruned TFLite Micro build for
*this* MNIST topology on Cortex-M4 likely lands somewhere between
80 KB and 150 KB; we haven't measured that locally yet. Unless and
until that measurement is in this table, treat the 105 KB row as a
useful reference, not a head-to-head benchmark.

Accuracy tested on the full MNIST test set (10,000 images) using
[`scripts/eval_full_mnist.py`](scripts/eval_full_mnist.py): f32 ONNX
98.82% vs. INT8-weights-simulated 98.85%, with 8 of 10,000 predictions
differing. **Methodology:** the "INT8-weights-simulated" path applies
edge-infer's exact per-tensor symmetric quantize-then-dequantize step
to each weight tensor and reruns the resulting model in ONNX Runtime
on the host. It is *not* a 10K-image QEMU run — that would take hours
and isn't currently scripted. The on-device Cortex-M4 binary can still
differ by a vanishingly small amount due to f32 accumulation order;
the spot-checked digits (`gen_test_data.py`) match within 1e-4. Rerun
the script to reproduce the exact disagreement count for your build
(small variation across toolchain versions is normal).

## How it works

```
  ONNX model
      |
      v
  1. Parse: read protobuf, extract graph ops + weight tensors
      |
      v
  2. Walk graph: propagate shapes, resolve names
      |
      v
  3. Emit Rust:
      - weights.rs: const arrays with baked-in weight values
      - ops.rs: generic operators (conv2d, relu, maxpool, dense)
      - lib.rs: predict() function chaining ops in graph order
      - Cargo.toml: no_std crate config with size optimization
```

The code generator (`edge_infer.py`) is a single Python script (~1.5 kLOC).
A Rust rewrite is planned -- single binary, no Python dependency.

A few honest implementation notes for the embedded ML reviewer:

- **`conv2d` has a per-iteration bounds check** inside the `kh, kw` loop
  to handle padding. The textbook win is to split into `border` /
  `interior` loops where the interior path skips the bounds check
  entirely, or precompute valid `(kh, kw)` ranges per `(oh, ow)`. The
  v0 kernel doesn't do this — it's on the optimization roadmap.
- **INT8 dequantizes inside the inner accumulator loop**
  (`sum += input[...] * (kernel[...] as f32 * scale)`). This means INT8
  in v0 is a flash-density win (4× more weights fit), not a runtime win
  vs. f32 — every weight is dequantized to f32 before MAC. CMSIS-NN
  integer-domain MAC is the obvious next step but isn't in v0.
- **Per-shape monomorphization is real.** Each unique combination of
  const generics on `conv2d` / `dense` produces a fresh function in
  the binary. For the bundled examples this is a non-issue (all 3×3
  convs); for a model with many distinct conv shapes the flash savings
  from compile-time embedding can be partly offset by code-size growth.
  We haven't seen this be a problem in practice yet, but it's something
  to measure if your model has a wide variety of layer shapes.

## Supported ops

| ONNX Op | Generated Rust | Notes |
|---|---|---|
| Conv | `ops::conv2d` | 2D convolution with const generics. **Stride must be 1** — stride>1 is hard-blocked at `--check`. |
| Relu | `ops::relu_inplace` | In-place via `[T]::as_flattened_mut` |
| MaxPool | `ops::max_pool2d` | **Only 2×2 stride-2** — other shapes are hard-blocked at `--check`. |
| Gemm | `ops::dense` | Dense / linear layer with bias; handles transB=0 and transB=1 |
| MatMul | `ops::dense` (no bias) | Same kernel as Gemm with a zero-bias path |
| Add | inlined | Element-wise add (used for Conv+bias and skip connections) |
| Reshape | `[T]::as_flattened()` | Zero-cost — reinterprets the same memory, no copy |
| Flatten | `[T]::as_flattened()` | Same kernel as Reshape |

Eight ops. Covers CNN classifiers (Conv→ReLU→MaxPool chains), pure
MLPs, and Conv+ReLU+Add residuals — i.e., the architectures the
six bundled examples exercise. Op support is demand-driven; more
ops ship as models demand them.

> Tip: export PyTorch models in `eval()` mode so `BatchNormalization`
> folds into the preceding `Conv`. Otherwise BN appears in the ONNX
> graph and edge-infer rejects it.

## If your model doesn't compile

That's the most useful feedback edge-infer can get. Op support is **demand-driven** — the most-requested ops ship next, in order.

**Realistic expectation:** the v0 op set covers MNIST/CIFAR/MLP-style topologies. Most non-toy models will need at least one op edge-infer doesn't yet support — common first-bounce ops are `Softmax`, `GlobalAveragePool`, stride-2 conv, depthwise conv, and `BatchNormalization` (the last folds away if you `model.eval()` before export). The `--check` output tells you exactly which ops you're missing in seconds, and the most-requested 5–10 ops are likely to ship within ~4 weeks of launch. A bounced model with a filed op-request is *more* valuable to the project than a successful first run — it shapes the roadmap.

**First, run the compatibility check:**

```bash
uv run python edge_infer.py path/to/your/model.onnx --check
```

`--check` lists every op in your model, marks supported / not-yet-implemented / unknown, and gives a yes/no verdict. No code generation, no compilation — answers in a few seconds.

**Then file an issue** (whichever applies):

- [**Op request**](https://github.com/idan-ben-ami/edge-infer/issues/new?template=op-request.yml) — one issue per missing op. The queue is sorted by 👍 count, so reactions matter.
- [**Model didn't compile**](https://github.com/idan-ben-ami/edge-infer/issues/new?template=model-didnt-compile.yml) — bigger context: what model, what target MCU, what you're trying to ship.
- [**Hardware test report**](https://github.com/idan-ben-ami/edge-infer/issues/new?template=hardware-test-report.yml) — actually got it onto a real board? Please share. Real-hardware data drives the roadmap.

A note on tone: don't worry about polish. A 30-second issue ("BatchNorm, MobileNetV2, STM32F4") is more useful than a 30-minute one with full reproduction steps. The point is signal, not paperwork.

## Current limitations

- INT8 is weight-only: weights are quantized to i8 and dequantized at inference
  time. Full integer inference (faster on Cortex-M with CMSIS-NN) is planned.
- 8 ONNX ops (Conv, Relu, MaxPool, Gemm, MatMul, Add, Reshape, Flatten).
  Covers CNN classifiers, MLPs, and Conv+ReLU+Add residuals. Not yet:
  Softmax, GlobalAveragePool, BatchNormalization (export with `eval()`
  to fold), Sigmoid/Tanh, RNNs, attention, depthwise conv. Conv stride
  > 1 and non-2×2 MaxPool are hard-blocked by `--check`.
- Tested on six examples — MNIST, Fashion-MNIST, Iris, CWRU bearing
  fault detector, two CIFAR-10 variants. Image / tabular /
  fault-detection domains. CNN and pure-MLP topologies. More
  architectures still welcome.
- The code generator is Python. A Rust rewrite is planned for
  single-binary distribution with no Python dependency.

## Project structure

```
edge-infer/
├── edge_infer.py             <- the code generator
├── pyproject.toml            <- Python dependencies (managed by uv)
├── examples/
│   ├── mnist/                <- CNN, handwritten digits (lm3s6965evb)
│   ├── fashion_mnist/        <- CNN, clothing categories (lm3s6965evb)
│   ├── iris/                 <- pure-Gemm MLP, 4 features (lm3s6965evb)
│   ├── vibration_anomaly/    <- MLP, real CWRU bearing fault data (lm3s6965evb)
│   ├── cifar10_tiny/         <- shrunk CNN, fits 64 KB SRAM (lm3s6965evb)
│   └── cifar10_mps2/         <- full CNN, 16 MB SRAM target (mps2-an386)
│
│   Each example dir contains:
│     train.py            <- training script (PyTorch, exports ONNX)
│     <model>.onnx        <- pre-trained model (ships with repo)
│     generated/          <- edge-infer output, browsable on GitHub
└── .gitignore
```

## Roadmap

- ~~INT8 quantization~~ done (per-tensor symmetric, weight-only dequant)
- Full integer inference: INT8 activations + INT8 weights for speed on Cortex-M (CMSIS-NN)
- More ONNX ops: BatchNorm, AvgPool, Softmax, Add/Mul elementwise, depthwise conv
- Rust rewrite: single binary, no Python dependency, faster compilation
- Real hardware benchmarks: STM32, nRF52, ESP32-S3 cycle counts and power measurements
- Multi-target: RISC-V (ESP32-C3), Cortex-M0/M7, target-specific SIMD intrinsics
- Model optimization: pruning, operator fusion, constant folding

## License

Licensed under either of

- MIT license ([LICENSE](LICENSE))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
