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

## Flash size — TFLite Micro comparison (locally measured)

> **Claim in README:** for our specific MNIST topology, edge-infer's
> 54 KB INT8 binary is **~2.3× smaller** than TFLite Micro built with
> only the ops this model needs (124 KB), and **~6.8× smaller** than
> a TFLite Micro all-ops sanity binary (363 KB).

The previous README quoted "~447 KB all-ops" without a local build on
disk to back it. We rebuilt TFLite Micro from source on this machine,
locally, against the same Cortex-M4+fp target edge-infer ships, and
got a different number — closer to ~363 KB. The "~105 KB" / "~275 KB"
figures from the [TinyML Platforms Benchmarking 2021 paper](https://arxiv.org/abs/2112.01319)
were measured on nRF52840 with a *different* MNIST topology and should
not be treated as apples-to-apples; we now have our own pruned-resolver
build for **our** topology and report that instead.

### Headline numbers (all measured on Cortex-M4+fp, `-Os --gc-sections`)

| Approach | Flash (text + data) | Source |
|---|---|---|
| **edge-infer INT8** | **54 KB** (text=55,008, data=0) | This repo, `cargo build --bin minimal --features minimal --target thumbv7em-none-eabihf --release` + `arm-none-eabi-size`. |
| TFLite Micro pruned for our exact MNIST topology | **124 KB** (text=127,256, data=104) | TFLM SHA `51bee03b`, op resolver = `Conv2D + MaxPool2D + FullyConnected + Reshape + Shape + StridedSlice + Pack`, embedded INT8 model = 58 KB, runtime + 7 ops = 66 KB |
| TFLite Micro all-ops sanity build (every public `Add*`, 118 ops) | **363 KB** (text=371,273, data=108) | Same TFLM SHA / toolchain. Worst-case "I forgot to prune" baseline. |

### Honest ratios

- **vs the fair pruned build (A.3): edge-infer is ~2.3× smaller.** This
  is the ratio that survives a hostile HN comment. It's not 8×.
- vs the all-ops baseline: 6.8×. That's the bound on "how much TFLM
  bloat you'd see if you didn't prune the resolver" — useful as
  context, dishonest as a headline.

### Why the win is real (and why it's smaller than the marketing claimed)

Of the 124 KB pruned TFLM build:
- ~58 KB is the embedded `.tflite` model file itself (FlatBuffer-encoded).
- ~66 KB is TFLM's interpreter + 7 op kernels + flatbuffer reader +
  schema parser + tensor allocator.

edge-infer's 54 KB is, by contrast:
- ~51 KB const arrays of weights + scales + biases.
- ~3 KB of generated `predict()` + `ops::*` MAC loops.
- 0 bytes of interpreter, schema parser, allocator, or flatbuffer reader.

The win comes from eliminating the runtime infrastructure, not from
having "smaller weights." The right framing is "no interpreter, no
flatbuffer parser, no schema machinery" — not "8× smaller flash."

### Reproduce — A.3 (the apples-to-apples figure)

Full methodology (TFLM clone + Keras model train + .tflite convert +
op-resolver-pruned link + size measure) lives in
`~/projects/external/tflm-bench/` outside this repo (gitignored from
edge-infer; doesn't pollute the repo). End-to-end wall clock (after
TFLM clone + toolchain download): ~3 minutes. The methodology document
is `BENCHMARKS-TFLM-DRAFT.md` at this repo root, kept for review/audit
purposes; full reproduce instructions there.

Specifically:
1. Clone TFLM at SHA `51bee03b`, run `gmake -f tools/make/Makefile
   TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp microlite` to
   build the static library + download the bundled `arm-none-eabi-gcc
   14.3.Rel1` toolchain (1.5 MB `.a` artifact).
2. Train an architecturally-equivalent Keras MNIST model
   (`Conv2D(8,3x3) → MaxPool2D → Conv2D(16,3x3) → MaxPool2D → Flatten
   → Dense(64) → Dense(10)`), 5 epochs, INT8 quantize via
   `tf.lite.TFLiteConverter` with a 200-image representative dataset.
   Spot-checked accuracy: 97.65% on 2K test images (within the same
   97–99% band as edge-infer's INT8 path).
3. Generate `main_a3.cc` with `MicroMutableOpResolver<7>` registering
   exactly the 7 ops the .tflite uses.
4. Link with the same `-Os --specs=nano.specs --specs=nosys.specs
   -Wl,--gc-sections` flags TFLM uses for its own library.
5. `arm-none-eabi-size a3_pruned.elf` → text=127,256, data=104 → 124 KB.

### Caveats (honest)

- **Toolchain mismatch.** edge-infer's 54 KB was measured with
  `arm-none-eabi-gcc 15.2.0` (Homebrew, no newlib needed for Rust).
  TFLM's 124 KB / 363 KB were measured with `arm-none-eabi-gcc
  14.3.Rel1` (downloaded by TFLM's Makefile because Homebrew's 15.2.0
  doesn't ship newlib). Both are `-Os` release with LTO/`--gc-sections`;
  cross-compiler size variance for similar workloads is typically <10%.
  Pinning both builds to 14.3.Rel1 would close this caveat — on the
  post-launch list.
- **A.3 not executed on Cortex-M4.** The 124 KB TFLM binary links
  cleanly — every symbol resolves, all 7 op kernels and the
  interpreter / allocator / flatbuffer paths are referenced. The model
  itself was independently validated for correctness in Python with
  `tf.lite.Interpreter` (97.65% on 2K samples). We did *not* run
  the linked binary on QEMU or silicon; that's a "verify it runs"
  follow-up, ~1–2 hours of work, on the post-launch list.
- **No CMSIS-NN.** TFLM was built with `OPTIMIZED_KERNEL_DIR` unset
  (reference kernels only), matching edge-infer's reference-kernel
  codegen. CMSIS-NN typically grows the all-ops build slightly (adds
  an int8 fast-path alongside the reference path) and helps runtime
  speed more than flash size. A CMSIS-NN row is on the post-launch
  list.
- **Float comparison not measured.** This is INT8 vs INT8. edge-infer's
  f32 path (204 KB) would compare to a TFLM build with float reference
  kernels enabled (~A.3 + ~50 KB extra). Out of scope here.

---

## Peak stack usage (real, measured)

> **Claim in README:** the MNIST CNN uses ~47 KB of stack on Cortex-M4;
> other examples range from < 256 bytes (iris MLP) to ~72 KB (full-
> resolution CIFAR-10 on mps2-an386). The earlier "~28 KB" analytic
> estimate was wrong — see methodology note below.

### Measured high-water marks (post-B.25 stack-slot reuse codegen)

| Example | Stack high-water | Pre-B.25 | Reduction |
|---|---|---|---|
| mnist (CNN) | **38,788 bytes (~38 KB)** | 48,260 B | **-19.6%** |
| fashion_mnist (CNN) | **38,788 bytes (~38 KB)** | 48,260 B | **-19.6%** |
| iris (MLP) | **≤ 256 bytes** | ≤ 256 B | already at floor |
| vibration_anomaly (MLP) | **416 bytes** | 416 B | already minimal |
| cifar10_tiny (small CNN) | **16,924 bytes (~17 KB)** | 20,060 B | **-15.6%** |
| cifar10_mps2 (CNN, full-size) | **54,812 bytes (~54 KB)** | 73,436 B | **-25.4%** — *now fits 64 KB SRAM lm3s6965evb!* |

**B.25 win:** the full-size CIFAR-10 model previously needed the 4 MB
SRAM mps2-an386 board (74 KB stack > 64 KB lm3s6965evb SRAM). With
phase-block scoping in the generator, it drops to 54 KB and runs on
the 64 KB lm3s6965evb. Functionally verified end-to-end via QEMU.

The earlier "shrunk to fit" `cifar10_tiny` example remains as a demo
of architectural pre-pooling for memory-budget-tight scenarios, but
the underlying premise ("you must shrink CIFAR-10 to fit a 64 KB
MCU") no longer holds — you can run the full-size model with the
new codegen.

### How the optimization works

The generator now wraps Conv-Relu-Pool clusters (and similar
single-pass groups) in `let escape_var = { ... };` Rust blocks. Each
block's intermediate buffers (e.g. `conv1_out` 25 KB, alive only
inside its block) become eligible for stack-slot reuse once the block
ends, while the block's "escape" tensor (`pool1_out` 6 KB) remains in
the outer scope. Rust + LLVM with `opt-level = "z"`, LTO, and
`codegen-units = 1` coalesces disjoint-lifetime stack slots
automatically; nested blocks just give the compiler the lifetime
information it needs.

Zero new `unsafe` introduced. The 38 KB (vs theoretical 31 KB
two-buffer ping-pong floor) leaves ~7 KB on the table that an
unsafe-pool approach could recover; that's a v1.0 roadmap item.

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
