# TFLite Micro Benchmarks — Draft for Review

**Author:** Hardening agent session, 2026-04-29
**Status:** ready for Idan's review

This file replaces the unsupported "447 KB" all-ops claim and the cited
"~105 KB" pruned figure (which came from a 2021 paper on a different MNIST
topology) with locally-built numbers for the **specific** edge-infer MNIST
2-conv+2-dense model on **Cortex-M4 with FPU**.

Headline: **TFLite Micro pruned for our exact MNIST topology = ~124 KB
flash. Edge-infer INT8 = 54 KB flash. Edge-infer is ~2.3× smaller, not
8× smaller.** All-ops baseline is ~363 KB, smaller than the README's
447 KB but still a meaningful 6.8× ratio.

---

## Environment

| Item | Value |
|---|---|
| Host OS | macOS (Darwin 25.4.0, arm64) |
| arm-none-eabi-gcc | Arm GNU Toolchain 14.3.Rel1 (Build arm-14.174) 14.3.1 20250623 — downloaded by TFLM's own `arm_gcc_download.sh` to `tensorflow/lite/micro/tools/make/downloads/gcc_embedded/` |
| GNU make | 4.4.1 (`gmake` from Homebrew; system `/usr/bin/make` is 3.81 and is rejected by TFLM's Makefile) |
| Python | 3.13 (in venv at `~/projects/external/tflm-bench/.venv`) |
| TensorFlow | 2.21.0 |
| Keras | 3.14.0 |
| TFLite Micro git SHA | `51bee03bed4776f1de88dd87226ff8c260f88e3c` (`51bee03 Default to 64-bit accumulation for 16x8 Fully Connected without bias (#3522)`) |
| TFLite Micro clone path | `~/projects/external/tflite-micro` |
| Working dir for benches | `~/projects/external/tflm-bench` |
| Target | `cortex-m4+fp` (= Cortex-M4 with FPv4-SP, hard float, little-endian, thumb). This matches edge-infer's `thumbv7em-none-eabihf` Rust target. |

### Why the downloaded toolchain (not Homebrew's arm-none-eabi-gcc)

Homebrew ships `arm-none-eabi-gcc 15.2.0` but **not** newlib, so it can't
compile any code that includes `<cstddef>` etc. TFLM's Makefile downloads
the official Arm GNU Toolchain 14.3.Rel1 darwin-arm64 release (which
bundles newlib) into `tools/make/downloads/gcc_embedded/`. We use that
toolchain end-to-end so the comparison is apples-to-apples and
reproducible without extra host setup.

---

## A.2 — TFLite Micro all-ops, Cortex-M4 sanity binary

**What "all-ops" means here.** Modern TFLM (2024+) **removed
`AllOpsResolver`** — the canonical way to register every op is now to
call every `Add*` method on a `MicroMutableOpResolver`. The auto-generator
script `gen_main_a2.py` reads `micro_mutable_op_resolver.h`, extracts
every public `AddX(...)` method (excluding `AddBuiltin`/`AddCustom`),
and emits a `main_a2.cc` that registers all 118 of them. Each `AddX()`
references `Register_X()` symbols inside `libtensorflow-microlite.a`,
which the linker therefore cannot strip via `--gc-sections`.

This is the worst-case "every kernel linked" measurement that the
README's old "447 KB" claim was supposed to model.

**Build the TFLM library** (one-time):

```
cd ~/projects/external/tflite-micro
gmake -f tensorflow/lite/micro/tools/make/Makefile \
      TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp microlite -j4
```

Produces `gen/cortex_m_generic_cortex-m4+fp_default_gcc/lib/libtensorflow-microlite.a`
(1,545,716 bytes — this is a static archive; only referenced ops end up
in the final ELF).

**Generate main and build A.2:**

```
cd ~/projects/external/tflm-bench
python3 gen_main_a2.py    # writes main_a2.cc with 118 AddX() calls
bash build_a2.sh
```

`build_a2.sh` runs (verbatim):

```
arm-none-eabi-g++ \
  -fno-rtti -fno-exceptions -fno-threadsafe-statics -Wnon-virtual-dtor \
  -fno-unwind-tables -fno-asynchronous-unwind-tables \
  -ffunction-sections -fdata-sections -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
  -std=c++17 \
  -mcpu=cortex-m4 -mfpu=auto -mthumb -mfloat-abi=hard -mlittle-endian \
  -DTF_LITE_MCU_DEBUG_LOG -fomit-frame-pointer -funsigned-char \
  -DCPU_M4=1 -DARMCM4 \
  -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE='"ARMCM4.h"' \
  -Os \
  -I<TFLM> \
  -I<TFLM>/tensorflow/lite/micro/tools/make/downloads \
  -I<TFLM>/.../downloads/gemmlowp \
  -I<TFLM>/.../downloads/flatbuffers/include \
  -I<TFLM>/.../downloads/kissfft \
  -I<TFLM>/.../downloads/ruy \
  -I<TFLM>/.../downloads/cmsis/Cortex_DFP/Device/ARMCM4/Include \
  -I<TFLM>/.../downloads/cmsis/CMSIS/Core/Include \
  -I<GEN>/genfiles/ \
  -mcpu=cortex-m4 -mfpu=auto -mthumb -mfloat-abi=hard \
  --specs=nosys.specs --specs=nano.specs \
  -Wl,--gc-sections -Wl,-Map=a2.map \
  main_a2.cc \
  -o a2_all_ops.elf \
  -Wl,--start-group <GEN>/lib/libtensorflow-microlite.a -lm -Wl,--end-group
```

Compile flags are **identical** to the ones TFLM uses for
`libtensorflow-microlite.a` itself (extracted via
`gmake ... microlite -n` after deleting one obj file). Link flags use
`--specs=nosys.specs --specs=nano.specs` to provide newlib stubs and
nano-libc, and `--gc-sections` to strip unreferenced code (this is what
real firmware ships). The linker emits warnings like `_close is not
implemented` because the nosys stubs intentionally fail at runtime —
fine for a measurement-only binary; no system calls are actually made
in `main`.

**`arm-none-eabi-size a2_all_ops.elf`:**

```
   text    data     bss     dec     hex   filename
 371273     108    5144  376525   5becd   a2_all_ops.elf
```

**Headline number:** **363 KB** flash (text + data = 371,381 B = 362.7 KB,
rounding up to 363).

**Notes:**
- `OPTIMIZED_KERNEL_DIR` is **not** set, i.e. CMSIS-NN reference kernels are
  not used. CMSIS-NN typically *grows* code size in the all-ops case
  because it adds an additional optimized-int8 path alongside the
  reference path. Reference kernels are the conservative baseline.
- `BUILD_TYPE=release` is the default; `-Os` is in the flags.
- 118 ops = every public `Add*` method on `MicroMutableOpResolver`,
  including signal-processing ops (FFT, filter banks, etc.). Real apps
  never need all of these, which is why this is an upper bound.

---

## A.3 — TFLite Micro op-resolver-pruned for our MNIST 2-conv+2-dense

This is the **fair apples-to-apples** comparison.

### Generating the .tflite model

`train_mnist.py` defines a Keras model architecturally equivalent to
edge-infer's PyTorch `MnistCNN`:

```
Conv2D(8,  3x3, padding='same', relu) -> MaxPool2D(2)
Conv2D(16, 3x3, padding='same', relu) -> MaxPool2D(2)
Flatten -> Dense(64, relu) -> Dense(10)
```

Trained on standard MNIST, 5 epochs, 128 batch. Result:

```
f32 test accuracy: 0.9852         (expected band: 97-99%, ✓)
```

Converted with `tf.lite.TFLiteConverter` using a 200-image
representative dataset for full INT8 quantization
(`TFLITE_BUILTINS_INT8`, `inference_input_type=int8`,
`inference_output_type=int8`).

```
mnist_int8.tflite: 59416 bytes
```

INT8 accuracy spot-check on 2,000 test images (no XNNPACK on the host
interpreter): **97.65%** — matches the f32 model's accuracy band. (For
direct comparison: edge-infer's own INT8 path is 98.76% on a topology
with the same channel counts; the small gap is normal cross-framework
variance.)

A dynamic-range fallback was also produced (`mnist_dyn.tflite`,
59,800 B) but was *not* used for the binary build — INT8 is the
fair comparison against edge-infer's INT8 output.

### Ops the .tflite model actually uses

```
$ python -c "i=tf.lite.Interpreter(model_path='mnist_int8.tflite'); ..."
CONV_2D
MAX_POOL_2D
CONV_2D
MAX_POOL_2D
SHAPE
STRIDED_SLICE
PACK
RESHAPE
FULLY_CONNECTED
FULLY_CONNECTED
```

ReLU is **fused** into Conv2D and FullyConnected (standard TFLite
behaviour). The Keras 3 → TFLite converter expands `Flatten` into
`Shape -> StridedSlice -> Pack -> Reshape`; this is converter-induced
overhead, not a model-design choice. Idan's PyTorch path does not
incur this — it's a fact about the TFLite ecosystem.

### Op resolver registrations

```cpp
constexpr int kNumOps = 7;
static tflite::MicroMutableOpResolver<kNumOps> resolver;
resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddFullyConnected();
resolver.AddReshape();
resolver.AddShape();
resolver.AddStridedSlice();
resolver.AddPack();
```

### main.cc summary

- Embed the 59,416-byte .tflite as `alignas(16) const unsigned char g_model[]`
  via `mnist_model.cc` (generated by `gen_main_a2.py`-style hex-dumping).
- Allocate a 16 KB tensor arena.
- `GetModel(g_model)` → version check → resolver setup → `MicroInterpreter`
  → `AllocateTensors()` → zero the int8 input → `Invoke()` → touch the
  output to defeat dead-code elimination.

The interpreter, allocation, and `Invoke()` paths are *all* referenced,
so the linker keeps the full TFLM runtime + the 7 ops + their
dependencies (e.g. `cmsis_nn` is *not* pulled in here because we did
not set `OPTIMIZED_KERNEL_DIR=cmsis_nn` when building the library —
reference kernels only).

### Build command

`build_a3.sh` uses **the same flags as A.2** plus `mnist_model.cc`:

```
arm-none-eabi-g++ <same CFLAGS as A.2> <same LDFLAGS as A.2> \
  main_a3.cc mnist_model.cc \
  -o a3_pruned.elf \
  -Wl,--start-group <GEN>/lib/libtensorflow-microlite.a -lm -Wl,--end-group
```

### `arm-none-eabi-size a3_pruned.elf`

```
   text    data     bss     dec     hex   filename
 127256     104   17244  144604   234dc   a3_pruned.elf
```

**Headline number:** **124 KB** flash (text + data = 127,360 B = 124.4 KB).

Of which the embedded .tflite model is 59,416 B (58 KB) sitting in
`.rodata`/`.text`, so the **TFLM runtime + 7 ops** themselves account
for ~67,944 B (66 KB) of code.

### Does it run?

The binary links cleanly with all symbols resolved (`AllocateTensors`,
`Invoke`, all 7 op kernel symbols, the schema parser, the flatbuffer
reader). It is **not** executed in this report because doing so
requires either a Cortex-M4 device, a board-specific bring-up
(startup code + linker script + semihosting), or QEMU with semihosting
configured — all out of scope for a size-only measurement. The model
itself was independently validated for correctness in Python with
`tf.lite.Interpreter` (97.65% on 2K MNIST test images) before being
embedded as a byte array, so we know the .tflite is a well-formed,
runnable model. **If runtime verification on M4 is needed for the
launch, suggest a follow-up that wires in a `mlibc.specs` semihosting
stub and runs under `qemu-system-gnuarmeclipse` or similar.**

---

## Comparison summary (for Idan)

| Approach | Flash (text+data) | Source |
|---|---|---|
| edge-infer INT8 | **54 KB** (text=55,008, data=0) | `mnist_model_q8/target/thumbv7em-none-eabihf/release/minimal` (verified in this session) |
| TFLite Micro all-ops (A.2) | **363 KB** (text=371,273, data=108) | This work |
| TFLite Micro pruned for our MNIST (A.3) | **124 KB** (text=127,256, data=104) | This work |

**Honest ratios:**

- vs A.3 (the fair comparison): **edge-infer is ~2.3× smaller**
  (54 KB vs 124 KB; 0.43×). Of A.3's 124 KB, ~58 KB is the embedded
  .tflite model and ~66 KB is TFLM runtime + 7 ops.
- vs A.2 (every-op baseline): edge-infer is ~6.8× smaller (54 KB vs
  363 KB; 0.149×). This is the better number to lead with for the
  "TFLM-bloat" framing, but it's the *unfair* comparison — real apps
  prune.

### What this means for the README

The current README claim is:

> | TFLite Micro (all ops) | ~447 KB | (reference, large) |
> | TFLite Micro (op-resolver-pruned) | ~105 KB | nRF52840 figure (different MNIST topology) |
> | TFLite Micro (typical) | ~275 KB | same paper |
> | edge-infer INT8 | 54 KB | this repo |

**Honest replacements (recommended):**

| Approach | Flash | Source |
|---|---|---|
| TFLite Micro (all-ops, our build) | **363 KB** | This repo, Cortex-M4+fp, TFLM SHA `51bee03`, `arm-none-eabi-gcc 14.3.1`, `-Os --gc-sections` |
| TFLite Micro (pruned for **our** MNIST topology) | **124 KB** | Same toolchain; .tflite model 58 KB + TFLM runtime + 7 ops 66 KB |
| edge-infer INT8 | **54 KB** | Same Cortex-M4+fp target |

The headline framing should change from "**8× smaller**" to **~2× smaller
on a like-for-like INT8 MNIST** — and the framing for *why* it's smaller
should foreground "no interpreter, no flatbuffer parser, no schema
machinery; weights are baked in as Rust constants." The 6.8× number
against the all-ops baseline can stay as "vs the easy/lazy way to ship
TFLM," but should be honestly labelled as such.

The cited "~105 KB" pruned figure from the 2021 paper is **lower** than
our locally-built 124 KB number for *our* topology. Plausible reasons:
(a) the paper used a different MNIST architecture with fewer ops and
fewer parameters; (b) ARM compiler 5/Keil produces tighter code than
GNU 14.3 + nano newlib for some workloads. Either way, the published
figure does not apply to our topology and should not be cited as if it
did.

---

## Blockers / open questions

1. **Runtime validation on Cortex-M4.** A.3's `Invoke()` path is linked
   but not executed in this report. To close this fully, add a QEMU or
   on-device run that prints the output tensor and compares against the
   host-side `tf.lite.Interpreter` result. Estimated 1–2 hours of work
   if you're willing to pull in `qemu-system-arm` with a Cortex-M4 board
   model (e.g. `lm3s6965evb`), or 0 hours if you have a Nucleo-F4 or
   similar on hand.

2. **CMSIS-NN variant not measured.** edge-infer doesn't use CMSIS-NN, so
   the fair comparison is reference-kernel TFLM (what we built). But
   chip-vendor partners may ask "what about CMSIS-NN?". Adding a third
   row with `OPTIMIZED_KERNEL_DIR=cmsis_nn` in the library build would
   take ~10 min and give: A.3-cmsis-nn flash. Expected: pruned size
   stays similar or grows slightly (CMSIS-NN adds an int8 fast-path
   alongside the reference path; for tiny models, the wins are runtime
   speed not flash size).

3. **Float build of edge-infer not in this comparison.** The report
   compares INT8 to INT8. edge-infer also has an f32 path (`mnist_model/`);
   if a launch reader asks "what about float?", the analogous TFLM number
   would be ~A.3's flash + ~50 KB extra for the f32 model and float
   reference kernels. Out of scope for this PR.

4. **Header-cstd reproducibility caveat.** Homebrew's
   `arm-none-eabi-gcc 15.2.0` does not bundle newlib and cannot build
   any of this — the TFLM-downloaded 14.3.Rel1 toolchain is required.
   Anyone trying to reproduce on macOS without running TFLM's
   `arm_gcc_download.sh` first will fail with `cstddef: No such file`.
   Documented.

---

## Sanity-check log

```
1.  arm-none-eabi-gcc --version                                       # 15.2.0 from brew (no newlib)
2.  brew install gnu-make? -> not needed; gmake 4.4.1 already installed
3.  arm-none-eabi-size mnist_model_q8/.../minimal                     # 55008 0 0 → edge-infer baseline confirmed
4.  git clone --depth=1 tflite-micro -> ~/projects/external/tflite-micro
5.  git rev-parse HEAD                                                # 51bee03bed4776f1de88dd87226ff8c260f88e3c
6.  gmake microlite -j4 (TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp)
    -> downloads ARM GNU Toolchain 14.3.Rel1 darwin-arm64
    -> downloads CMSIS, flatbuffers, gemmlowp, ruy, kissfft, pigweed
    -> builds libtensorflow-microlite.a (1,545,716 B)
7.  python3.13 -m venv ~/projects/external/tflm-bench/.venv
    pip install tensorflow                                            # tf 2.21.0, keras 3.14.0
8.  python train_mnist.py                                             # f32 acc 0.9852, int8 .tflite 59416 B
9.  python -c "tf.lite.Interpreter on mnist_int8.tflite, 2000 samples" # int8 acc 0.9765
10. python gen_main_a2.py                                             # main_a2.cc with 118 AddX()
11. python ... gen mnist_model.cc                                      # 59416-byte tflite as alignas(16) const unsigned char[]
12. bash build_a2.sh                                                  # text 371273, data 108  -> 363 KB
13. bash build_a3.sh                                                  # text 127256, data 104  -> 124 KB
```

## Files referenced

All bench-side artifacts live **outside** the edge-infer repo at
`~/projects/external/tflm-bench/`:

- `gen_main_a2.py` — generates `main_a2.cc` from the resolver header
- `main_a2.cc` — generated all-ops main (118 AddX calls)
- `build_a2.sh` — A.2 build script
- `a2_all_ops.elf`, `a2.map` — A.2 outputs
- `train_mnist.py` — Keras model trainer + INT8 converter
- `mnist_int8.tflite`, `mnist_dyn.tflite` — converted models
- `mnist_model.cc` — the .tflite as a C array
- `main_a3.cc` — A.3 pruned main
- `build_a3.sh` — A.3 build script
- `a3_pruned.elf`, `a3.map` — A.3 outputs

To reproduce on a fresh checkout: re-run steps 4–13 in the sanity-check
log above. End-to-end wall clock (excluding TFLM clone + toolchain
download): ~3 minutes.
