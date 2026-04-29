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

## Methodology — per-recording / temporal-gap split

An earlier version of `train.py` used a **per-window random 80/20**
split *after* windowing. Because adjacent windows overlap by 50 %
(`HOP = WINDOW_SIZE / 2`), the random split placed near-duplicate
windows into both train and test. The model "learned" specific shifts
of recordings rather than generalizing the fault signature, and reported
a leakage-flattered 100 %. That methodology is gone.

The current `train.py` uses two separate split strategies, matching what
real fault-detection pipelines do:

1. **Per-recording split for FAULTY:** with 9 faulty recordings (3
   fault types × 3 severities), 2 entire recordings are held out for
   test. The choice is a seeded permutation, not cherry-picked — for
   `SEED = 42` the test set is `1797_B_7_DE12.npz` (ball fault, 0.007″
   defect) and `1797_IR_7_DE12.npz` (inner-race, 0.007″). The model
   never sees a single sample from these recordings during training.
2. **Temporal-gap split for HEALTHY:** we have only 1 healthy recording.
   Per-recording split is not possible; instead, the first 80 % of the
   recording is train and the last 20 % is test, with a 1-window gap
   (1024 samples, ≈ 85 ms) between them so no train window overlaps any
   test window.

### Result on the fair split

```
Test Accuracy (PyTorch): 100.00%
Test Accuracy (ONNX f32): 100.00%
Anomaly precision: 1.000 | recall: 1.000 | F1: 1.000
Confusion matrix:  TP=473  FP=0  FN=0  TN=94
```

This is **genuine generalization** — to unseen recordings, with a
non-overlapping healthy temporal split — not the leakage number it
replaced. The Cortex-M4 INT8 binary classifies the bundled demo
sample (a held-out faulty window from the test set) correctly.

### What this result is *not*

A few honest caveats about what 100% on the fair split does and
doesn't prove:

- The held-out faulty severities are 0.007″, the **smallest** defect.
  Training contains 0.014″ and 0.021″ of the same fault types. So we're
  measuring generalization across severities of the same fault type,
  not generalization to entirely **novel fault types** (the harder
  direction).
- All 9 faulty recordings come from one of three fault types
  (Ball / Inner-Race / Outer-Race@6). A truly novel fault type
  (cage failure, lubricant degradation, mixed faults) would be a
  different test we haven't run.
- The healthy recording is single-source (one motor, one load
  condition: 1797 RPM, drive-end @ 12 kHz). Performance on a
  different motor / RPM / sensor placement is unknown.
- The MLP may be picking up "any non-quiet vibration spectrum" rather
  than discriminating fault classes. Binary healthy-vs-faulty is a
  much easier task than multi-class fault-type identification.

Treat the 100 % number as evidence the **pipeline is working**, not as
evidence the model is production-ready. A real predictive-maintenance
deployment would also evaluate on out-of-distribution motors, novel
fault classes, and concept-drift over time.

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
