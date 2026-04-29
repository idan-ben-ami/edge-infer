"""Train a small CIFAR-10 CNN and export to ONNX for edge-infer validation.

CIFAR-10 is 32x32x3 RGB. Same op set as MNIST/Fashion-MNIST examples
(Conv-Relu-Pool stacks + Dense layers, all stride-1 convs and 2x2 stride-2
maxpools, no BatchNorm) — just deeper because the input is 3x larger and
in colour. Demonstrates edge-infer scaling beyond 1-channel grayscale.

Architecture (all activations stay under ~30 KB to fit lm3s6965evb's 64 KB RAM):
  Conv(3->8, 3x3, pad=1)   -> Relu -> MaxPool 2x2  : 32x32x3 -> 16x16x8
  Conv(8->16, 3x3, pad=1)  -> Relu -> MaxPool 2x2  : 16x16x8 -> 8x8x16
  Conv(16->32, 3x3, pad=1) -> Relu -> MaxPool 2x2  : 8x8x16  -> 4x4x32
  Flatten + FC(512->64) -> Relu -> FC(64->10)
"""

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

# CIFAR-10 classes, in label order:
# 0=airplane  1=automobile  2=bird  3=cat  4=deer
# 5=dog       6=frog        7=horse 8=ship 9=truck


class CifarCNN(nn.Module):
    """Small CIFAR-10 CNN — only ops in edge-infer's supported set."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(torch.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.pool(torch.relu(self.conv3(x)))   # 8x8   -> 4x4
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    # CIFAR-10 per-channel normalization stats
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = SCRIPT_DIR / 'data'
    train_dataset = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cpu')
    model = CifarCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    num_epochs = 8
    for epoch in range(1, num_epochs + 1):
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
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.1f}%")

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

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    onnx_path = str(SCRIPT_DIR / "cifar_cnn.onnx")

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

    # PyTorch 2.x dynamo exporter splits weights into an external .data file
    # by default. Re-save as a single self-contained ONNX file to match
    # examples/mnist/mnist_cnn.onnx (one-file shippable format).
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(onnx_model, onnx_path, save_as_external_data=False)
    data_path = SCRIPT_DIR / "cifar_cnn.onnx.data"
    if data_path.exists():
        data_path.unlink()

    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully (single-file format)")


if __name__ == "__main__":
    train()
