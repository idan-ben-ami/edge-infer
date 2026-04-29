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

## Peak activation RAM

> **Claim in README:** the MNIST CNN's peak activation RAM is ~28 KB
> (analytic estimate); other examples range < 1 KB (MLPs) to ~44 KB
> (full-resolution CIFAR-10).

_Status: **analytic only** — pending Phase A.5 (stack-paint measurement
on QEMU). Numbers in the README's example table are computed by summing
the largest live activation buffers at any single layer of `predict()`._

### Analytic numbers per example

| Example | Peak buffer (live activation tensors) | Source |
|---|---|---|
| mnist | ~28 KB (input 3 KB + conv1_out 25 KB) | analytic |
| fashion_mnist | ~28 KB | same topology as mnist |
| iris | < 1 KB | analytic |
| vibration_anomaly | < 1 KB | analytic |
| cifar10_tiny | ~30 KB | analytic |
| cifar10_mps2 | ~44 KB | analytic |

These numbers do **not** include stack frame overhead (saved registers,
return addresses). On real silicon, expect another 1–4 KB of stack
above the activation buffers.

### Real measurement (Phase A.5 deliverable)

_TODO: once A.5 lands, paste the QEMU-measured stack high-water for
each example here, with the reproduce command._

---

## Inference cycle count

> **Claim in README:** _no claim yet — currently silent on speed._

_Status: **pending** — Phase A.6. We'll measure cycles via the Cortex-M4
DWT cycle counter, run on QEMU, and report cycles per `predict()` call.
QEMU's cycle counter is approximate (not bit-accurate vs silicon), but
useful for relative comparisons (e.g. before/after the conv2d
border/interior split in B.8)._

### Last measured output

_TODO: A.6 deliverable._

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
