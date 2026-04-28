"""Train a TINY CIFAR-10 CNN that fits in lm3s6965evb's 64 KB SRAM.

The standard cifar10 example (channels 8->16->32) hangs in QEMU because the
first conv produces 8x32x32 f32 = 32 KB, plus 12 KB input, plus all later
intermediate buffers (~82 KB total live), overflowing 64 KB SRAM.

This tiny variant pre-pools the RGB input to 16x16 BEFORE the first conv,
which collapses every downstream buffer by 4x. Estimated peak RAM:
    input         3x32x32 f32 = 12 KB
    pool0_out     3x16x16 f32 =  3 KB
    conv1_out     8x16x16 f32 =  8 KB
    pool1_out     8x8x8   f32 =  2 KB
    conv2_out    16x8x8   f32 =  4 KB
    pool2_out    16x4x4   f32 =  1 KB
    fc1_out      32       f32 = 128 B
    fc2_out      10       f32 =  40 B
                              = ~30 KB total intermediates + 12 KB input = ~42 KB
Leaves ~22 KB headroom for cortex-m runtime + stack.

Architecture (all stride-1 Conv2d, all 2x2 stride-2 MaxPool, no BatchNorm):
  MaxPool 2x2                                : 3x32x32 -> 3x16x16   (downsample)
  Conv(3->8, 3x3, pad=1)   -> Relu -> MaxPool: 3x16x16 -> 8x8x8
  Conv(8->16, 3x3, pad=1)  -> Relu -> MaxPool: 8x8x8   -> 16x4x4
  Flatten + FC(256->32) -> Relu -> FC(32->10)
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).parent

# CIFAR-10 classes, in label order:
# 0=airplane  1=automobile  2=bird  3=cat  4=deer
# 5=dog       6=frog        7=horse 8=ship 9=truck


class CifarTinyCNN(nn.Module):
    """RAM-constrained CIFAR-10 CNN — only ops in edge-infer's supported set."""

    def __init__(self):
        super().__init__()
        # Input is pre-pooled at the start of forward() to halve spatial dims.
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)                           # 32x32x3 -> 16x16x3 (downsample input)
        x = self.pool(torch.relu(self.conv1(x)))   # 16x16   -> 8x8
        x = self.pool(torch.relu(self.conv2(x)))   # 8x8     -> 4x4
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
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
    model = CifarTinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    num_epochs = 6
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

    # Re-save as a single self-contained ONNX file (post-merge external data).
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
