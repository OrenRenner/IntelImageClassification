import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class Classificator:

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = torchvision.models.resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 6))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = torchvision.transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model.to(self.device)

    def __call__(self, image: Image.Image):
        image_torch = self.transform(image)
        image_torch = image_torch.unsqueeze(0)
        image_torch = image_torch.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_torch)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()
