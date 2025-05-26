import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class EnsembleNet(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleNet, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()

        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Identity()

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 2048 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1 = self.resnet50(x)
        x2 = self.resnet101(x)
        x3 = self.efficientnet(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.classifier(x)

def build_model():
    return EnsembleNet(num_classes=7)
