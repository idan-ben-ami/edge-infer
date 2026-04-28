"""Evaluate the MNIST CNN on the full 10,000-image test set, comparing
f32 ONNX inference to the same model with edge-infer's INT8 quantization
applied to its weights.

This reproduces the README's "INT8 differs from f32 on a small handful
of samples" claim. The exact disagreement count varies slightly between
toolchain / pytorch / numpy versions (different rounding paths in the
ONNX export); rerun this script to get the number for your build.

Run:
    uv sync --extra train
    uv run python scripts/eval_full_mnist.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

try:
    from torchvision import datasets, transforms
except ImportError:
    print("Error: torchvision is required. Install with: uv sync --extra train")
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
MNIST_DIR = REPO_ROOT / "examples" / "mnist"
ONNX_PATH = MNIST_DIR / "mnist_cnn.onnx"


def quantize_tensor_symmetric(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Same per-tensor symmetric int8 routine as edge_infer.py."""
    abs_max = float(np.max(np.abs(arr)))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_dequantize(arr: np.ndarray) -> np.ndarray:
    """Round-trip through int8 to see what the on-device weights actually look like."""
    q, scale = quantize_tensor_symmetric(arr.astype(np.float32))
    return (q.astype(np.float32) * scale).astype(np.float32)


def make_int8_simulated_model(src_path: Path) -> bytes:
    """Return a serialized ONNX model with weight tensors replaced by their
    quantize-then-dequantize equivalents. Biases left untouched, matching
    edge_infer.py's policy."""
    m = onnx.load(str(src_path), load_external_data=True)

    bias_dims = {1}  # 1D bias tensors are not quantized
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32:
            continue
        if arr.ndim in bias_dims:
            continue
        new_arr = quantize_dequantize(arr)
        new_init = numpy_helper.from_array(new_arr.astype(arr.dtype), name=init.name)
        init.CopyFrom(new_init)
    return m.SerializeToString()


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(str(MNIST_DIR / "data"), train=False, download=True, transform=transform)
    images = np.zeros((len(ds), 1, 28, 28), dtype=np.float32)
    labels = np.zeros(len(ds), dtype=np.int64)
    for i, (img, lbl) in enumerate(ds):
        images[i] = img.numpy()
        labels[i] = int(lbl)
    return images, labels


def run(session: ort.InferenceSession, images: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    preds = np.empty(len(images), dtype=np.int64)
    batch = 256
    for i in range(0, len(images), batch):
        out = session.run(None, {name: images[i:i + batch]})[0]
        preds[i:i + len(out)] = out.argmax(axis=1)
    return preds


def main() -> None:
    if not ONNX_PATH.exists():
        raise SystemExit(f"missing {ONNX_PATH}; run examples/mnist/train.py first")

    print(f"Loading test set from {MNIST_DIR / 'data'} ...")
    images, labels = load_test_set()
    print(f"  {len(images)} images")

    print("Running f32 ONNX inference on full test set ...")
    sess_f32 = ort.InferenceSession(str(ONNX_PATH))
    preds_f32 = run(sess_f32, images)

    print("Building INT8-weights-simulated model and re-running ...")
    int8_bytes = make_int8_simulated_model(ONNX_PATH)
    sess_q = ort.InferenceSession(int8_bytes)
    preds_q = run(sess_q, images)

    acc_f32 = float((preds_f32 == labels).mean())
    acc_q = float((preds_q == labels).mean())
    disagreements = int((preds_f32 != preds_q).sum())

    print()
    print(f"f32 accuracy:                       {acc_f32 * 100:.4f}%  ({(preds_f32 == labels).sum()}/{len(labels)})")
    print(f"INT8-weights accuracy:              {acc_q * 100:.4f}%  ({(preds_q == labels).sum()}/{len(labels)})")
    print(f"f32 vs INT8-weights disagreements:  {disagreements} of {len(labels)}")
    print()
    print("Note: the int8 simulation here applies the same quantize-then-dequantize")
    print("step the generated Rust crate does to its weight constants. The")
    print("on-device run can still differ by a vanishingly small amount due to")
    print("Cortex-M f32 accumulation order vs. ONNX Runtime's; the disagreement")
    print("count above is the Python-side simulation upper bound.")


if __name__ == "__main__":
    main()
