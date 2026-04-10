import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class PoopModel:
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(1280, 2)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image):
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return predicted.item(), confidence.item()