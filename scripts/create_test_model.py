#!/usr/bin/env python3
"""Create a simple ONNX model for testing airML."""

import torch
import torch.nn as nn
import os

class SimpleModel(nn.Module):
    """Simple CNN for image classification (10 classes)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    # Create model
    model = SimpleModel()
    model.eval()

    # Dummy input (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX using dynamo=False (legacy export, single file)
    output_path = "models/simple_cnn.onnx"

    # Remove old files
    for f in [output_path, output_path + ".data"]:
        if os.path.exists(f):
            os.remove(f)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,  # Use legacy export (single file)
    )

    print(f"Model saved to {output_path}")
    print(f"Input shape: [1, 3, 224, 224]")
    print(f"Output shape: [1, 10]")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
