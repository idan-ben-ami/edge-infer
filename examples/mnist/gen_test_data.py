"""Generate reference test data from the exported ONNX model for Rust validation.

Requires the 'train' extra: uv sync --extra train
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from torchvision import datasets, transforms
except ImportError:
    print("Error: torchvision is required. Install with: uv sync --extra train")
    raise SystemExit(1)

SCRIPT_DIR = Path(__file__).parent


def format_f32(val: float) -> str:
    """Format a float for Rust source code."""
    return f"{val:.8e}_f32"


def generate_test_data():
    # Load ONNX model
    session = ort.InferenceSession(str(SCRIPT_DIR / "mnist_cnn.onnx"))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX model loaded - input: {input_name}, output: {output_name}")

    # Load MNIST test set (same normalization as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(str(SCRIPT_DIR / 'data'), train=False, transform=transform)

    # Pick one sample per digit (0-9)
    digit_samples = {}
    for img, label in test_dataset:
        if label not in digit_samples:
            digit_samples[label] = img.numpy()  # shape: (1, 28, 28)
        if len(digit_samples) == 10:
            break

    # Stack in digit order
    test_inputs = np.array([digit_samples[d] for d in range(10)], dtype=np.float32)
    print(f"Test inputs shape: {test_inputs.shape}")  # (10, 1, 28, 28)

    # Run inference
    test_outputs = session.run(
        [output_name],
        {input_name: test_inputs}
    )[0]
    print(f"Test outputs shape: {test_outputs.shape}")  # (10, 10)

    # Print predictions
    predictions = np.argmax(test_outputs, axis=1)
    print(f"Predictions: {predictions.tolist()}")
    print(f"Expected:    {list(range(10))}")
    correct = sum(p == e for p, e in zip(predictions, range(10)))
    print(f"Correct: {correct}/10")

    # Save numpy files
    np.save(str(SCRIPT_DIR / "test_inputs.npy"), test_inputs)
    np.save(str(SCRIPT_DIR / "test_outputs.npy"), test_outputs)
    print("\nSaved test_inputs.npy and test_outputs.npy")

    # Generate Rust test data file (single test case: first image = digit 0)
    idx = 0
    img = test_inputs[idx]       # (1, 28, 28)
    out = test_outputs[idx]      # (10,)
    digit = int(predictions[idx])

    lines = []
    lines.append("// Auto-generated test data from MNIST CNN ONNX model")
    lines.append("// Input: normalized MNIST image, Output: logits (10 classes)")
    lines.append("")

    # Format input as [[[f32; 28]; 28]; 1]
    lines.append("pub const TEST_INPUT: [[[f32; 28]; 28]; 1] = [")
    for c in range(img.shape[0]):  # channels (1)
        lines.append("    [")
        for row in range(img.shape[1]):  # 28 rows
            vals = ", ".join(format_f32(img[c, row, col]) for col in range(img.shape[2]))
            lines.append(f"        [{vals}],")
        lines.append("    ],")
    lines.append("];")
    lines.append("")

    # Format output as [f32; 10]
    out_vals = ", ".join(format_f32(v) for v in out)
    lines.append(f"pub const EXPECTED_OUTPUT: [f32; 10] = [{out_vals}];")
    lines.append("")

    lines.append(f"pub const EXPECTED_DIGIT: usize = {digit};")
    lines.append("")

    rust_code = "\n".join(lines)
    with open(SCRIPT_DIR / "test_data.rs", "w") as f:
        f.write(rust_code)
    print(f"Saved test_data.rs (digit={digit})")

    # Generate Rust test data file with ALL 10 test cases
    all_lines = []
    all_lines.append("// Auto-generated test data from MNIST CNN ONNX model")
    all_lines.append("// All 10 test cases (one per digit, 0-9)")
    all_lines.append("")
    all_lines.append("pub const NUM_TESTS: usize = 10;")
    all_lines.append("")

    # Format inputs as [[[[f32; 28]; 28]; 1]; 10]
    all_lines.append("pub const TEST_INPUTS: [[[[f32; 28]; 28]; 1]; 10] = [")
    for i in range(10):
        img = test_inputs[i]  # (1, 28, 28)
        all_lines.append("    [")
        for c in range(img.shape[0]):  # channels (1)
            all_lines.append("        [")
            for row in range(img.shape[1]):  # 28 rows
                vals = ", ".join(format_f32(img[c, row, col]) for col in range(img.shape[2]))
                all_lines.append(f"            [{vals}],")
            all_lines.append("        ],")
        all_lines.append("    ],")
    all_lines.append("];")
    all_lines.append("")

    # Format outputs as [[f32; 10]; 10]
    all_lines.append("pub const EXPECTED_OUTPUTS: [[f32; 10]; 10] = [")
    for i in range(10):
        out_vals = ", ".join(format_f32(v) for v in test_outputs[i])
        all_lines.append(f"    [{out_vals}],")
    all_lines.append("];")
    all_lines.append("")

    # Expected digits
    digits_str = ", ".join(str(int(predictions[i])) for i in range(10))
    all_lines.append(f"pub const EXPECTED_DIGITS: [usize; 10] = [{digits_str}];")
    all_lines.append("")

    all_rust_code = "\n".join(all_lines)
    with open(SCRIPT_DIR / "test_data_all.rs", "w") as f:
        f.write(all_rust_code)
    print(f"Saved test_data_all.rs (all 10 digits)")


if __name__ == "__main__":
    generate_test_data()
