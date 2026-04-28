# CIFAR-10 tiny

A deliberately shrunk CIFAR-10 CNN sized to fit in the **64 KB SRAM**
of `lm3s6965evb`. Companion to `examples/cifar10_mps2/`, which runs the
full-size variant on a board with 4 MB SRAM.

## Why MaxPool is the first op

The architecture is unusual: a 2×2 MaxPool sits **before** the first
Conv2d. That isn't an aesthetic choice — it's the only way to fit the
network in 64 KB without dropping accuracy further.

The standard CIFAR-10 toy CNN (channels 8 → 16 → 32) overflows 64 KB at
the first conv: an 8×32×32 f32 activation tensor alone is 32 KB; plus
the 12 KB input; plus every later intermediate; total live memory is
~82 KB. By pre-pooling 32×32 RGB → 16×16 RGB *before* any conv, every
downstream buffer collapses 4×:

| Buffer | Size |
|--------|------|
| Input (3×32×32 f32) | 12 KB |
| `MaxPool0` out (3×16×16 f32) | 3 KB |
| `Conv1` out (8×16×16 f32) | 8 KB |
| `MaxPool1` out (8×8×8 f32) | 2 KB |
| `Conv2` out (16×8×8 f32) | 4 KB |
| `MaxPool2` out (16×4×4 f32) | 1 KB |
| `Gemm1` out (32 f32) | 128 B |
| `Gemm2` out (10 f32) | 40 B |
| **Peak intermediates + input** | **~30 KB** |

Leaving ~22 KB for the cortex-m runtime + stack, which fits.

The cost is accuracy: pre-pooling drops the spatial resolution to 16×16
before any feature is learned, costing ~13 percentage points vs. the
full-resolution model (53.5 % here vs. 67 % in `cifar10_mps2`). The
example is not meant to be a competitive CIFAR-10 model — it's the
biggest image classifier we could squeeze into the smallest realistic
embedded target, end-to-end through edge-infer.

## Run

```bash
cd generated
cargo run --bin demo --features demo --release
```

Or in QEMU (semihosting):

```bash
cargo build --bin demo --features demo --target thumbv7em-none-eabihf --release
qemu-system-arm -cpu cortex-m4 -machine lm3s6965evb -nographic \
    -semihosting-config enable=on,target=native \
    -kernel target/thumbv7em-none-eabihf/release/demo
```

## See also

- `examples/cifar10_mps2/` — same network family, full resolution, on a
  4 MB-SRAM board. The honest production deployment shape.
