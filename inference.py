"""
inference.py - SageMaker inference handler for CIFAR-10 CNN classifier
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def model_fn(model_dir):
    """Load the trained CIFAR-10 CNN model."""
    model = CIFAR10CNN()
    model_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✅ CIFAR-10 CNN Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Preprocess image for CIFAR-10 model."""
    try:
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        print(f"Received image size: {image.size}")

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        tensor = transform(image).unsqueeze(0)
        print(f"Final tensor shape: {tensor.shape}")
        return tensor

    except Exception as e:
        print(f"❌ Error in input_fn: {str(e)}")
        raise


def predict_fn(input_data, model):
    """Run prediction."""
    with torch.no_grad():
        output = model(input_data)
        prediction = int(output.argmax(dim=1).item())
        confidence = torch.softmax(output, dim=1)[0][prediction].item()
        
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"Predicted: {class_names[prediction]} (confidence: {confidence:.2%})")
        return {"prediction": class_names[prediction], "confidence": round(confidence, 4)}


def output_fn(prediction, accept):
    return json.dumps(prediction), accept
