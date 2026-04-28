# Vibration anomaly — CWRU bearing-fault MLP

A 32→64→32→2 MLP trained on the public **CWRU Bearing Fault** dataset
(Case Western Reserve University) to classify vibration windows as
healthy vs faulty. Demonstrates edge-infer on real, public, non-toy
fault-detection data — what a predictive-maintenance pipeline on a
microcontroller would actually look like at the inference end.

## Pipeline

1. Download the CWRU drive-end-bearing recordings (12 kHz sampling) from
   the public mirror at `srigas/CWRU_Bearing_NumPy`. Healthy vs. several
   inner-race / outer-race / ball faults; faulty classes are merged.
2. Slice each recording into 1024-sample windows (~85 ms @ 12 kHz) with
   50% overlap (`HOP=512`).
3. Featurize each window: FFT magnitude + `log1p` compression to keep
   faulty-class feature magnitudes inside the INT8-quantization-friendly
   range. Reduce 512 FFT bins → 32 binned magnitudes.
4. Standardize per-feature using statistics fit on the **healthy-train**
   split only — realistic for production, where only healthy samples are
   reliably available at calibration time.
5. Train MLP, export to ONNX, generate INT8 Rust crate via edge-infer.

## Honest accuracy caveat (window-overlap leakage)

The training split is **per-window random 80/20**, *after* the 50 %-
overlap windowing in step (2). Adjacent windows share 512 of their 1024
samples, so the random split places near-duplicates in train and test.
The reported 100 % test accuracy reflects this leakage, not the
generalization a fielded predictive-maintenance system would see.

The pipeline is honest about *running* on real recordings; it is **not
honest about** the difficulty of the classification task as scored.
A fairer split would partition by **recording file** (each motor /
load / fault condition belongs to exactly one of train or test) before
windowing — the expected drop is to roughly 90–95 % depending on fault
class coverage.

This example exists to show that the edge-infer pipeline produces a
working INT8 Cortex-M binary on real public data. Use the per-recording
split before trusting any number this generates.

## Why log1p

Without `log1p` on the FFT magnitudes, faulty-bearing windows have
energy 100–1000× larger than healthy windows in the dominant bins. INT8
per-tensor symmetric quantization on the first `Gemm` layer collapses
the healthy-window magnitudes into the same code, and the sequential
MAC loop on Cortex-M4 accumulates catastrophic precision loss. `log1p`
compresses the dynamic range so both classes share a usable scale.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Downloads CWRU data, builds features, trains MLP, exports ONNX. |
| `cwru_bearing_mlp.onnx` | Exported f32 model (~14 KB on disk). |
| `test_samples.npz` | Held-out test windows + labels (used by `demo_input.rs`). |
| `demo_input.rs` | One real test window picked from `test_samples.npz`. |
| `generated/` | INT8 Rust crate produced by `edge_infer.py`. |

## Run

```bash
uv run python examples/vibration_anomaly/train.py
uv run python edge_infer.py \
    examples/vibration_anomaly/cwru_bearing_mlp.onnx \
    --output examples/vibration_anomaly/generated/ \
    --quantize int8

cd examples/vibration_anomaly/generated
cargo run --bin demo --features demo --release
```

Expected output: `Predicted class: 1 (FAULTY)`.
