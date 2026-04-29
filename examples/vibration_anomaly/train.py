"""Train a tiny MLP on REAL bearing-fault accelerometer data and export ONNX.

Dataset: CWRU Bearing Dataset (Case Western Reserve University), repackaged
as .npz by https://github.com/srigas/CWRU_Bearing_NumPy (MIT-licensed
re-distribution of the original CWRU data, which is publicly released for
academic/research use by Case Western).

Source URL:
  https://github.com/srigas/CWRU_Bearing_NumPy
  raw files at:
    https://raw.githubusercontent.com/srigas/CWRU_Bearing_NumPy/main/Data/<RPM>/<file>.npz

Why this repo: original CWRU is .mat (needs scipy). The srigas mirror ships
each Matlab file as a .npz with float64 accelerometer arrays under keys
'DE' (drive-end) and 'FE' (fan-end). We use only DE @ 12 kHz so we don't
need scipy at all.

Pipeline:
  1. Download a small curated set of CWRU files (1 normal + several faulty
     conditions: ball, inner-race, outer-race) at 1797 RPM, 12 kHz DE.
  2. Slice the raw signal into 1024-sample windows (~85 ms @ 12 kHz) with
     50% overlap.
  3. Featurize: rfft magnitude (513 bins) -> drop DC -> 512 bins -> avg-pool
     to 32 bins. Same trick as the synthetic example, deterministic on MCU.
  4. Standardize per-feature using stats fit on the *normal* split only
     (realistic fault-detection setup).
  5. Train MLP 32->64->32->2 (binary: healthy vs faulty), CE loss, weighted.
  6. Export to ONNX with dynamo=False, merge external data into one file.

Architecture: pure MLP -- Flatten + Gemm + Relu only -- so it stays inside
the 8 ops edge-infer supports.
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "data"
CACHE_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 1024     # samples per FFT window (~85 ms at 12 kHz)
HOP = 512              # 50% overlap
N_FEATURES = 32        # FFT-bin count after average-pool
SAMPLE_RATE = 12000    # Hz, CWRU drive-end @ 12 kHz
SEED = 42

# Curated subset of CWRU 1797 RPM, drive-end @ 12 kHz. The "Normal" file is
# the healthy baseline; the rest are different fault types and severities.
BASE_URL = (
    "https://raw.githubusercontent.com/srigas/CWRU_Bearing_NumPy/main/Data/1797%20RPM/"
)
NORMAL_FILE = "1797_Normal.npz"
FAULTY_FILES = [
    # (filename, fault-type tag) -- all DE12 (drive-end @ 12 kHz)
    "1797_IR_7_DE12.npz",     # inner race, 0.007" defect
    "1797_IR_14_DE12.npz",    # inner race, 0.014" defect
    "1797_IR_21_DE12.npz",    # inner race, 0.021" defect
    "1797_B_7_DE12.npz",      # ball, 0.007"
    "1797_B_14_DE12.npz",     # ball, 0.014"
    "1797_B_21_DE12.npz",     # ball, 0.021"
    "1797_OR@6_7_DE12.npz",   # outer race @6 o'clock, 0.007"
    "1797_OR@6_14_DE12.npz",  # outer race @6, 0.014"
    "1797_OR@6_21_DE12.npz",  # outer race @6, 0.021"
]


def _download(name: str) -> Path:
    """Cache a single .npz file locally so reruns are free."""
    local = CACHE_DIR / name.replace("@", "AT")
    if local.exists():
        return local
    url = BASE_URL + name.replace("@", "%40")
    print(f"  fetching {name}...")
    blob = urllib.request.urlopen(url, timeout=60).read()
    local.write_bytes(blob)
    return local


def _load_de(name: str) -> np.ndarray:
    """Load drive-end accelerometer channel from a CWRU npz, as 1-D float32."""
    local = _download(name)
    arr = np.load(local)
    return arr["DE"].astype(np.float32).reshape(-1)


def _windows(signal: np.ndarray, win: int = WINDOW_SIZE, hop: int = HOP) -> np.ndarray:
    """Stride-trick non-copy windowing."""
    n = (len(signal) - win) // hop + 1
    if n <= 0:
        return np.empty((0, win), dtype=signal.dtype)
    out = np.zeros((n, win), dtype=signal.dtype)
    for i in range(n):
        out[i] = signal[i * hop : i * hop + win]
    return out


def _featurize(windows: np.ndarray) -> np.ndarray:
    """rfft magnitude -> drop DC -> avg-pool -> log1p compression.

    The log1p compression is critical for INT8 inference: raw FFT magnitudes
    of faulty bearings are 100-1000x larger than healthy ones. Standardizing
    by healthy-only stats then yields z-scores in the thousands, which
    overwhelms f32 precision in the per-output sequential MAC loop on
    Cortex-M4 (LTO + FMA reordering loses too many bits when intermediate
    sums dwarf the per-MAC delta). log1p(magnitude) compresses both classes
    into a similar dynamic range while preserving discriminative info.
    """
    spec = np.abs(np.fft.rfft(windows, axis=1))[:, 1:]  # (N, 512)
    pooled = spec.reshape(spec.shape[0], N_FEATURES, -1).mean(axis=2)
    return np.log1p(pooled).astype(np.float32)


def build_dataset():
    """Build train / test splits with the **fair** methodology.

    Why this matters: an earlier version of this script windowed *all*
    recordings together, then random-80/20-split the resulting window
    list. Because adjacent windows overlap by 50% (HOP = WINDOW_SIZE/2),
    the random split placed near-duplicate windows into both train and
    test. The model "learned" to recognize specific shifts of a recording
    rather than to generalize the fault signature, and reported 100%
    test accuracy. That's leakage, not generalization.

    The fix is two-pronged, matching what production fault-detection
    pipelines actually do:

      1. **Per-recording split for FAULTY:** with 9 faulty recordings
         (3 fault types × 3 severities), hold out 2 entire recordings
         for test. The model never sees a single sample from a held-out
         recording during training. Concretely: for the 9-recording
         pool, the test set is whichever 2 a seeded permutation picks.

      2. **Temporal split for HEALTHY:** we have only 1 healthy
         recording, so per-recording split is impossible. Instead, take
         the first 80% of the recording for train and the last 20% for
         test, with a 1-window gap (1024 samples ≈ 85 ms) between them
         to ensure no train window overlaps any test window.

    Expected drop vs the leakage-prone split: ~85-95% test accuracy
    instead of 100%. The 100% number was the inflated upper bound;
    this is the honest one.
    """
    print("Loading CWRU bearing data...")
    print(f"  healthy: {NORMAL_FILE}")
    healthy_sig = _load_de(NORMAL_FILE)
    print(f"    raw samples: {len(healthy_sig)}")

    # Temporal split on the single healthy recording: first 80% train,
    # last 20% test, with a 1-window gap to kill boundary overlap.
    n_total = len(healthy_sig)
    cut_train_end = int(0.8 * n_total) - WINDOW_SIZE   # leave gap of 1 window
    cut_test_start = cut_train_end + WINDOW_SIZE       # gap = WINDOW_SIZE
    healthy_train_sig = healthy_sig[:cut_train_end]
    healthy_test_sig = healthy_sig[cut_test_start:]
    Xh_tr = _featurize(_windows(healthy_train_sig))
    Xh_te = _featurize(_windows(healthy_test_sig))
    print(f"    healthy train windows (first 80% of recording): {Xh_tr.shape[0]}")
    print(f"    healthy test  windows (last 20% of recording):  {Xh_te.shape[0]}")

    # Per-recording split for faulty: hold out 2 entire recordings for test.
    # Seeded permutation makes the choice deterministic; no cherry-picking.
    print("  faulty:")
    faulty_files_perm = list(FAULTY_FILES)
    np.random.default_rng(SEED).shuffle(faulty_files_perm)
    n_test_faulty_files = 2
    test_faulty_files = faulty_files_perm[:n_test_faulty_files]
    train_faulty_files = faulty_files_perm[n_test_faulty_files:]
    print(f"    test recordings (held out, never seen during training): {test_faulty_files}")
    print(f"    train recordings: {train_faulty_files}")

    Xf_tr_list = []
    for f in train_faulty_files:
        sig = _load_de(f)
        feats = _featurize(_windows(sig))
        print(f"      train  {f}: {feats.shape[0]} windows")
        Xf_tr_list.append(feats)
    Xf_tr = np.concatenate(Xf_tr_list, axis=0)

    Xf_te_list = []
    for f in test_faulty_files:
        sig = _load_de(f)
        feats = _featurize(_windows(sig))
        print(f"      test   {f}: {feats.shape[0]} windows")
        Xf_te_list.append(feats)
    Xf_te = np.concatenate(Xf_te_list, axis=0)

    print(
        f"\nTotal windows: "
        f"healthy={Xh_tr.shape[0]}+{Xh_te.shape[0]}, "
        f"faulty={Xf_tr.shape[0]}+{Xf_te.shape[0]}"
    )
    rng = np.random.default_rng(SEED)

    # Standardize on healthy-train stats (realistic: only healthy data
    # is reliably available at calibration time in production).
    mean = Xh_tr.mean(axis=0)
    std = Xh_tr.std(axis=0) + 1e-6

    def norm(x):
        return ((x - mean) / std).astype(np.float32)

    X_train = np.concatenate([norm(Xh_tr), norm(Xf_tr)], axis=0)
    y_train = np.concatenate(
        [np.zeros(len(Xh_tr), dtype=np.int64), np.ones(len(Xf_tr), dtype=np.int64)]
    )
    X_test = np.concatenate([norm(Xh_te), norm(Xf_te)], axis=0)
    y_test = np.concatenate(
        [np.zeros(len(Xh_te), dtype=np.int64), np.ones(len(Xf_te), dtype=np.int64)]
    )

    # Shuffle train/test
    p_tr = rng.permutation(len(X_train))
    p_te = rng.permutation(len(X_test))
    return X_train[p_tr], y_train[p_tr], X_test[p_te], y_test[p_te], mean, std


class VibrationMLP(nn.Module):
    """Pure MLP: 32 -> 64 -> 32 -> 2.

    Flatten + 3 Gemm + 2 Relu. No conv, no pool, no batchnorm -- keeps the
    op set inside what edge-infer supports today.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(N_FEATURES, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train, X_test, y_test, mean, std = build_dataset()
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    print(f"\nTrain: {len(X_train)} ({n_pos} faulty / {n_neg} healthy)")
    print(f"Test:  {len(X_test)} ({(y_test == 1).sum()} faulty)")

    # Reshape to (N, 1, 1, 32) so the ONNX input has rank 4 like the original
    # synthetic example, exercising the static-shape Flatten op.
    X_train_t = torch.from_numpy(X_train).reshape(-1, 1, 1, N_FEATURES)
    X_test_t = torch.from_numpy(X_test).reshape(-1, 1, 1, N_FEATURES)
    train_ds = TensorDataset(X_train_t, torch.from_numpy(y_train))
    test_ds = TensorDataset(X_test_t, torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cpu")
    model = VibrationMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Inverse-frequency class weighting.
    class_w = torch.tensor(
        [1.0, max(n_neg, 1) / max(n_pos, 1)], dtype=torch.float32
    )
    criterion = nn.CrossEntropyLoss(weight=class_w)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:2d}/10 - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%")

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.append(pred.numpy())
            all_targets.append(target.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = 100.0 * (preds == targets).mean()
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"\nTest Accuracy (PyTorch): {acc:.2f}%")
    print(f"Anomaly precision: {precision:.3f} | recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # --- Export to ONNX ---
    dummy_input = torch.randn(1, 1, 1, N_FEATURES, device=device)
    onnx_path = str(SCRIPT_DIR / "cwru_bearing_mlp.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    print(f"\nExported ONNX model to {onnx_path}")

    # Merge any external weight data into a single self-contained .onnx.
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    data_path = SCRIPT_DIR / "cwru_bearing_mlp.onnx.data"
    if data_path.exists():
        data_path.unlink()
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully (single-file format)")

    # Sanity-check ONNX vs PyTorch + report ONNX-runtime test accuracy.
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(
        None, {"input": X_test_t.numpy()}
    )[0]
    ort_preds = ort_logits.argmax(axis=1)
    ort_acc = 100.0 * (ort_preds == targets).mean()
    print(f"Test Accuracy (ONNX f32): {ort_acc:.2f}%")

    # Save normalization stats and a few samples for the embedded demo.
    np.savez(
        SCRIPT_DIR / "test_samples.npz",
        mean=mean,
        std=std,
        X_test=X_test[:32],
        y_test=y_test[:32],
    )

    # Pick a real held-out faulty sample with high model confidence, and
    # write it into demo_input.rs for the QEMU run.
    faulty_idx_pool = np.where(y_test == 1)[0]
    faulty_logits = ort_logits[faulty_idx_pool]
    # most-confident faulty sample (largest class-1 margin)
    margins = faulty_logits[:, 1] - faulty_logits[:, 0]
    chosen = int(faulty_idx_pool[int(np.argmax(margins))])
    sample = X_test[chosen]
    sample_logits = ort_logits[chosen]
    print(
        f"\nDemo sample: test idx={chosen}, true label={int(y_test[chosen])}, "
        f"ONNX logits={sample_logits.tolist()}"
    )

    sample_str = ", ".join(f"{v:.6}_f32" for v in sample.tolist())
    demo_input_rs = SCRIPT_DIR / "demo_input.rs"
    demo_input_rs.write_text(
        "// Auto-generated by train.py -- one real CWRU faulty bearing sample\n"
        "// (post-FFT, post-standardization) so the QEMU demo prints a\n"
        "// meaningful predicted class instead of using all-zero input.\n"
        f"pub static SAMPLE: [[[f32; {N_FEATURES}]; 1]; 1] = "
        f"[[[{sample_str}]]];\n"
        f"pub const EXPECTED_CLASS: usize = {int(y_test[chosen])};\n"
    )
    print(f"Wrote faulty demo sample to {demo_input_rs}")


if __name__ == "__main__":
    train()
