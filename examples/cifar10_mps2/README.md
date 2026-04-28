# CIFAR-10 on QEMU `mps2-an386`

Same 39 K-parameter CIFAR-10 CNN as `examples/cifar10/`, but compiled for
a Cortex-M4 board with **enough RAM to actually run it in QEMU**.

The default lm3s6965evb (used by every other edge-infer demo) only has
64 KB of SRAM. The CIFAR-10 INT8 model needs ~44 KB of activation memory
in the worst-case Conv layer plus the usual stack, so it hangs there.
This example switches to QEMU's `mps2-an386` machine — **same Cortex-M4F
core, but 4 MB FLASH + 4 MB SRAM** — which probes whether edge-infer
scales beyond the smallest demo board.

## Files

| File | Purpose |
|------|---------|
| `cifar_cnn.onnx` | Model from `examples/cifar10/` (copied verbatim). |
| `train.py` | Training script (copied verbatim from `examples/cifar10/`). |
| `demo_input.rs` | One real CIFAR-10 test sample (a "ship", class 8) extracted from the test batch and per-channel-normalized to match `train.py`. The same file is also placed at `generated/src/bin/demo_input.rs` and included by the demo. |
| `generated/` | Fresh INT8 crate, with **modified `memory.x` and `.cargo/config.toml`** for `mps2-an386`. |

## Memory map (MPS2-AN386)

From QEMU `hw/arm/mps2.c`:

| Region | Origin | Length |
|--------|--------|--------|
| FLASH (ZBT SSRAM1, code) | `0x00000000` | 4 MB |
| RAM (ZBT SSRAM2, data)   | `0x20000000` | 4 MB |

This is encoded in `generated/memory.x`. Compare to the `lm3s6965evb`
default (256 KB FLASH, 64 KB RAM) used by other examples.

## How to build & run

```bash
# from this directory:
cd generated
cargo build --bin demo --features demo --target thumbv7em-none-eabihf --release

# Option 1 — `cargo run` will use the configured runner in
# `generated/.cargo/config.toml`:
cargo run --bin demo --features demo --release

# Option 2 — invoke QEMU directly:
qemu-system-arm \
    -cpu cortex-m4 \
    -machine mps2-an386 \
    -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/demo
```

## Expected output

```
edge-infer CIFAR-10 demo  (target: QEMU mps2-an386, Cortex-M4)
True class: 8 (ship)
Predicted class: 8 (ship)
Correct!
Logits:
  [0] airplane   2.1228
  [1] automobile 3.7352
  ...
  [8] ship       5.2207
  [9] truck      1.0757
```

The Cortex-M4F INT8 logits match the f32 ONNX reference within ~0.04 on
the winning class — quantization noise, not a bug.

## Why a different board?

`lm3s6965evb` has 64 KB SRAM. The CIFAR-10 model's worst Conv layer needs
~44 KB just for one activation tensor; add the stack and the
secondary buffer and you're past 64 KB. `mps2-an386` has 4 MB SRAM,
same Cortex-M4F core, so we keep the `thumbv7em-none-eabihf` target and
just lift the RAM ceiling.

This is also the realistic story: a production embedded ML target board
(STM32H7, NXP RT1170, etc.) typically has 512 KB – 2 MB of SRAM. Demoing
on `mps2-an386` is closer to that regime than the 64 KB toy board.
