import torch
from torch import nn
from transformers import PreTrainedModel
from .configuration import CustomModelConfig
from torchvision import transforms
from PIL import Image
import sys

class CustomModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CustomModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * (input_shape[1] // 16) * (input_shape[2] // 16), 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CustomClassifier(PreTrainedModel):
    config_class = CustomModelConfig


    def __init__(self, config):
        super().__init__(config)
        self.model = CustomModel(config.input_size, config.num_classes)
        self.preprocess = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['cat', 'dog']

    def forward(self, x):
        try:
            x = Image.open(x).convert("RGB")
        except Exception as e:
            raise Exception(f"Error: Unable to load image file {x}. Check if the file exists or is in the right format. Details: {e}")

        x = self.preprocess(x).unsqueeze(0)
        

        return self.model(x)
    
    def predict(self, x, get_class=False):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            if not get_class:
                return {
                    "cat": round(probabilities[0][0].item(), 3),
                    "dog": round(probabilities[0][1].item(), 3)
                }
            else:
                
                return self.classes[probabilities.argmax(dim=1).item()]
        


