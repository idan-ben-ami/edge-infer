"""Train a small MNIST CNN and export to ONNX for edge-infer validation."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Pin all RNGs so accuracy reproduces byte-identical across runs.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SCRIPT_DIR = Path(__file__).parent

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)   # 8 filters
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)   # 16 filters
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = SCRIPT_DIR / 'data'
    train_dataset = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(str(data_dir), train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model, optimizer, loss
    device = torch.device('cpu')  # Keep on CPU for reproducibility
    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training loop
    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
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
        print(f"Epoch {epoch}/5 - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%")

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_acc = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    onnx_path = str(SCRIPT_DIR / "mnist_cnn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"\nExported ONNX model to {onnx_path}")

    # PyTorch 2.x splits weights into an external .data file by default for
    # any model bigger than ~1 KB. Re-save as a single self-contained ONNX
    # file so the repo ships a one-file model that loads on a fresh clone
    # without a sidecar.
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    data_path = SCRIPT_DIR / "mnist_cnn.onnx.data"
    if data_path.exists():
        data_path.unlink()

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully (single-file format)")


if __name__ == "__main__":
    train()
