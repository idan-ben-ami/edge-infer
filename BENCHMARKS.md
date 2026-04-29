# edge-infer — Benchmarks & Reproducibility

> Every quantitative claim in `README.md` should be reproducible from this
> document in ≤ 5 minutes. If you find a number in the README that doesn't
> have a section here, that's a bug — open an issue.
>
> This file is the **single source of truth** for benchmarks. Other docs
> (blog post, launch-day posts, gate1-launch.md) cite numbers from here.

**Last full reproduction sweep:** _pending — see HARDENING-PLAN.md_

---

## Environment

The numbers below were measured on:

| Component | Version |
|---|---|
| Host OS | macOS 15 (Darwin 25.4.0, arm64 / Apple Silicon) |
| Rust toolchain | rustc 1.95.0 (2026-04-14) |
| ARM GCC (size measurement) | arm-none-eabi-gcc 15.2.0 |
| QEMU | QEMU 11.0.0 |
| Python | 3.11 (via uv 0.7.12) |
| ONNX | onnx 1.21.0, onnxruntime 1.25.0 |

Any reader trying to reproduce should pin to the same major versions.
Differences within a major version are usually fine; cross-major-version
differences (e.g. rustc 1.95 → 1.96) can shift binary sizes by a few
hundred bytes but should not move headline numbers.

---

## Flash size — edge-infer (the headline)

> **Claim in README:** the INT8 MNIST CNN compiles to ~54 KB of flash on
> Cortex-M4F.

### Reproduce

```bash
# From a fresh clone
uv sync
uv run python edge_infer.py examples/mnist/mnist_cnn.onnx \
    --output ./mnist_model/ --quantize int8

cd mnist_model
cargo build --bin minimal --features minimal \
    --target thumbv7em-none-eabihf --release

arm-none-eabi-size target/thumbv7em-none-eabihf/release/minimal
```

### Last measured output (2026-04-29, post-A.1)

```
$ uv run python edge_infer.py examples/mnist/mnist_cnn.onnx \
    --output /tmp/test_minimal/ --quantize int8
$ cd /tmp/test_minimal
$ cargo build --bin minimal --features minimal \
    --target thumbv7em-none-eabihf --release
$ arm-none-eabi-size target/thumbv7em-none-eabihf/release/minimal
   text	   data	    bss	    dec	    hex	filename
  55008	      0	      0	  55008	   d6e0	target/thumbv7em-none-eabihf/release/minimal
```

→ **55,008 bytes (~54 KB)** of flash for the minimal production binary.
Independently reproducible from any fresh clone. `data + bss = 0`
because all weights are baked into `text` as `const` arrays.

### What "minimal" means

The `minimal` binary uses `panic-halt` and no semihosting — i.e., what
you'd actually flash to a real MCU. Compile flags: `opt-level="z"`, LTO,
full release, `codegen-units = 1`. The companion `demo` binary used for
QEMU output adds ~21 KB of `cortex-m-semihosting` + `panic-semihosting`;
that's debugging convenience, not the firmware-shippable size.

---

## Flash size — TFLite Micro comparison

> **Claim in README:** edge-infer is meaningfully smaller than TFLite
> Micro on Cortex-M4 for our specific MNIST topology.

_Status: **pending** — see Phase A.2 + A.3 in `HARDENING-PLAN.md`.
Independent agent session (HARDENING-AGENT-A2-A3-PROMPT.md) is doing
the local TFLM build. Numbers here will be filled in from
`BENCHMARKS-TFLM-DRAFT.md` once that agent's work is merged._

The README currently quotes:
- 447 KB (claimed local all-ops build) — **unverified**
- ~105 KB (cited from [arxiv 2112.01319](https://arxiv.org/abs/2112.01319),
  measured on nRF52840 with a *different* MNIST topology)
- ~275 KB (same paper)

Until A.2/A.3 land:
- 447 KB is **not** independently reproducible from this repo.
- 105/275 KB are **cited only**; not apples-to-apples for our topology.

The fair, defensible apples-to-apples row will be A.3's measurement of
TFLM with `MicroMutableOpResolver` registering only the ops our MNIST
topology uses.

---

## Peak stack usage (real, measured)

> **Claim in README:** the MNIST CNN uses ~47 KB of stack on Cortex-M4;
> other examples range from < 256 bytes (iris MLP) to ~72 KB (full-
> resolution CIFAR-10 on mps2-an386). The earlier "~28 KB" analytic
> estimate was wrong — see methodology note below.

### Measured high-water marks

| Example | Stack high-water | Reproduce |
|---|---|---|
| mnist (CNN) | **48,260 bytes (~47 KB)** | `cargo run --bin stack_probe --features demo --release` on lm3s6965evb |
| fashion_mnist (CNN) | **48,260 bytes (~47 KB)** | same as mnist (same topology) |
| iris (MLP) | **≤ 256 bytes** | < 256 reads as 256 due to safety-margin floor |
| vibration_anomaly (MLP) | **416 bytes** | exact |
| cifar10_tiny (small CNN) | **20,060 bytes (~20 KB)** | exact |
| cifar10_mps2 (CNN) | **73,436 bytes (~72 KB)** | mps2-an386 only — exceeds lm3s6965evb's 64 KB SRAM, which is why this example needs the bigger board |

### Reproduce

```bash
cd examples/mnist/generated
cargo build --bin stack_probe --features demo \
    --target thumbv7em-none-eabihf --release
qemu-system-arm -cpu cortex-m4 -machine lm3s6965evb -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/stack_probe
```

Expected output:

```
Stack high-water mark: 48260 bytes
(measured by stack-painting; excludes 1 KB safety margin at top)
predicted class: 1
```

### Methodology

The `stack_probe` binary uses the cortex-m-rt `#[pre_init]` hook to
paint the entire free stack region (from `__ebss` up to a 256-byte
safety offset below `_stack_start`) with the marker `0xDEADBEEF`
*before* `main` runs. After `predict()` returns, the binary walks the
stack region from the bottom upward and finds the lowest address whose
value is no longer the marker — that's the high-water mark, the lowest
SP reached during execution. `_stack_start - high_water` is the stack
usage.

The 256-byte safety margin at the top exists because `pre_init` itself
runs on the stack and would clobber its own frame if we painted too
far up. For models whose `predict()` uses less than 256 bytes of stack
(the two MLPs), this measurement reports 256 as a floor — the real
number is somewhere ≤ 256.

QEMU emulates RAM correctly, so this measurement matches what a real
Cortex-M4 would report when flashed with the same binary. (Unlike
cycle counting — see below.)

### Why the analytic estimate was wrong

The README originally said "~28 KB peak activation RAM" for mnist,
computed as the largest pair of live activation buffers at any single
layer (`input` 3 KB + `conv1_out` 25 KB = ~28 KB).

That assumed Rust would reuse stack slots for `let mut` activation
buffers whose lifetimes don't overlap. **It doesn't.** Even with
`opt-level = "z"`, LTO, and `codegen-units = 1`, the compiler keeps
every `let mut` buffer alive for the full function. So the real stack
footprint is the **sum** of all activation buffers:

| Buffer | Size |
|--------|------|
| `conv1_out` | 25,088 B |
| `pool1_out` | 6,272 B |
| `conv2_out` | 12,544 B |
| `pool2_out` | 3,136 B |
| `fc1_out` | 256 B |
| `fc2_out` | 40 B |
| Subtotal | 47,336 B |
| Frames + saved regs (overhead) | ~924 B |
| **Total measured** | **48,260 B** |

(`INPUT` is `static`, not on the stack, so it doesn't count.)

48 KB on the lm3s6965evb's 64 KB SRAM is **75% of available memory**.
That's tight; a real deployment would want either a slightly bigger
MCU (most production STM32F4/F7/H7 / nRF52 / ESP32 parts have ≥128 KB)
or a code-gen optimization to reuse stack slots across non-overlapping
buffer lifetimes (post-launch work; see HARDENING-PLAN.md roadmap).

---

## Inference cycle count

> **Claim in README:** _none yet — and the QEMU emulator can't give us
> a defensible one. Real silicon is required._

### Status: blocked by QEMU

The generator emits a `cycles` binary that uses the Cortex-M4 DWT
cycle counter (`CYCCNT` register) to time `predict()` execution. The
sequence is the textbook:

```rust
cp.DCB.enable_trace();
cp.DWT.enable_cycle_counter();
let _ = predict(&INPUT);                    // warmup, discard
let start = DWT::cycle_count();
let output = predict(&INPUT);
let end = DWT::cycle_count();
let cycles = end.wrapping_sub(start);
```

This works on real Cortex-M4 silicon (STM32F4, nRF52, etc.). On
`qemu-system-arm` with `-cpu cortex-m4 -machine lm3s6965evb` (or
`mps2-an386`), `CYCCNT` returns 0 — QEMU's TCG-based emulation does
not tick the DWT cycle counter. We tested with `-icount shift=0` as
well; same result. There's no QEMU plugin for cortex-m cycle counting
in the version we ship with (QEMU 11.0.0).

### Reproduce on real hardware

```bash
cd examples/mnist/generated
cargo build --bin cycles --features demo \
    --target thumbv7em-none-eabihf --release
# Flash to a real Cortex-M4 board (e.g. STM32F411 Nucleo, nRF52840 DK)
# and connect via probe-rs / OpenOCD with semihosting enabled.
# Expected output: "predict cycles (DWT, QEMU approximation): NNNNN"
```

### Reproduce on QEMU (will report 0 — known limitation)

```bash
cd examples/mnist/generated
cargo build --bin cycles --features demo \
    --target thumbv7em-none-eabihf --release
qemu-system-arm -cpu cortex-m4 -machine lm3s6965evb -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/cycles
```

Output:
```
predict cycles (DWT, QEMU approximation): 0
predicted class: 1
```

The "0" is not edge-infer's bug — DWT_CYCCNT isn't ticked on
qemu-system-arm. The binary itself is correct and ships in every
generated example; running it on silicon gives you a real number.
Plan to ship results from a physical board before v1.0.

---

## Accuracy — full test-set evaluation

> **Claim in README:** MNIST INT8 accuracy is 98.76% (5 of 10,000
> predictions differ from f32 baseline of 98.78%).

### Reproduce

```bash
uv sync --extra train
uv run python scripts/eval_full_mnist.py
```

### Last measured output (2026-04-29, post-A.4 with `SEED = 42`)

```
f32 accuracy:                       98.8200%  (9882/10000)
INT8-weights accuracy:              98.8500%  (9885/10000)
f32 vs INT8-weights disagreements:  8 of 10000
```

Note: INT8-weights accuracy (98.85%) is fractionally *higher* than f32
(98.82%) here — quantization noise can flip wrong predictions to right
ones in edge cases. This is normal at this scale and not evidence INT8
is "better"; both numbers are within 0.05% of each other.

### Methodology

The script:
1. Loads the full MNIST 10K test set via torchvision.
2. Runs the f32 ONNX model through ONNX Runtime on the host.
3. Builds a quantize-then-dequantize copy of the same ONNX (applying
   edge-infer's exact per-tensor symmetric INT8 quantization to weight
   tensors only; biases left in f32).
4. Runs that copy through ONNX Runtime.
5. Reports both top-1 accuracies and the disagreement count.

This is **not** a 10K-image QEMU run — that would take hours and isn't
currently scripted. The on-device Cortex-M4 binary can still differ
from this Python simulation by a vanishingly small amount due to f32
accumulation order, but the script's "INT8 differs from f32 on 5 of
10,000" upper bound is what edge-infer's INT8 quantization actually
costs you on this model.

For per-class confusion matrix and calibration plots, see Phase C.19.

---

## Per-example accuracy

| Example | Accuracy | Notes |
|---|---|---|
| mnist (CNN) | INT8 98.85% / f32 98.82% | Verified 2026-04-29 (A.4). `SEED = 42`. |
| fashion_mnist (CNN) | 89.04% | Verified 2026-04-29 (A.4). `SEED = 42`. Mediocre vs. literature (LeNet-style hits 91-93%); 5 epochs is undercooked — improving this is a post-launch task. |
| iris (MLP) | 96.67% | Verified 2026-04-29 (A.4). `SEED = 42`. Down from a previous "100%" that was a lucky seed; this is the honest result. |
| vibration_anomaly (MLP) | 100% on per-recording fair split (567 test windows, 0 errors) | Verified 2026-04-29 (A.7). `SEED = 42`. 2 entire fault recordings held out + temporal-gap healthy split. See `examples/vibration_anomaly/README.md` for honest caveats. |
| cifar10_tiny (small CNN) | 54.56% | Verified 2026-04-29 (A.4). `SEED = 42`. Limit demo (model is shrunk to fit 64 KB SRAM); honest accuracy. |
| cifar10_mps2 (CNN) | 63.96% | Verified 2026-04-29 (A.4). `SEED = 42`. Production-shape model on 4 MB SRAM target. Below literature (small-CNN CIFAR-10 hits ~75%); 10 epochs is undercooked — see post-launch roadmap. |

Reproduce any of them with:

```bash
uv sync --extra train
uv run python examples/<name>/train.py
```

After A.4, every `train.py` will pin its random seed for byte-identical
reproducibility.

---

## Comparison vs onnx2c

_Status: **pending** — Phase B.11. onnx2c is the natural baseline (same
compile-time-embed approach, different language). Result will land here
once a head-to-head local build is done._

---

## Monomorphization / binary growth on diverse-shape models

_Status: **pending** — Phase B.13. We'll measure binary growth for a
synthetic model with 12 distinct conv shapes vs 12 same-shape convs to
quantify the monomorphization concern Rust embedded reviewers raise._

---

## How to reproduce *everything*

For a full sweep that lands every number above on a fresh clone:

```bash
git clone https://github.com/idan-ben-ami/edge-infer.git
cd edge-infer
uv sync --extra train

# 1. Generate the canonical Rust crate from the bundled MNIST ONNX
uv run python edge_infer.py examples/mnist/mnist_cnn.onnx \
    --output ./mnist_model/ --quantize int8

# 2. Flash size (the 54 KB headline)
cd mnist_model
cargo build --bin minimal --features minimal \
    --target thumbv7em-none-eabihf --release
arm-none-eabi-size target/thumbv7em-none-eabihf/release/minimal
cd ..

# 3. Accuracy (98.78 / 98.76 / 5 of 10K)
uv run python scripts/eval_full_mnist.py

# 4. Re-run any example training
uv run python examples/mnist/train.py        # any of the 6
```

For TFLite Micro / onnx2c / cycle counts / stack peak measurements,
see the dedicated sections above and the corresponding tasks in
`ideas/edge-infer/HARDENING-PLAN.md`.

---

## Change log

(Append-only.)

- 2026-04-29 — file created (Phase A.0). Skeleton; most sections marked
  _pending_ with cross-references to HARDENING-PLAN.md tasks. The
  98.76/98.78/5-of-10K accuracy section and the historical 54 KB
  legacy measurement are the only fully-populated sections at creation.
