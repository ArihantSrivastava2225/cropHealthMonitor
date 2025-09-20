"""
CNN Feature Extractor using a pre-trained ResNet-34 model, adapted for
10-channel satellite and texture data.
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.utils.checkpoint as checkpoint

class ResNetEncoder(nn.Module):
    def __init__(self, feature_vector_size=128):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        original_conv1 = resnet.conv1
        
        # Adapt the first layer for 10-channel input
        new_conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy original weights for the first 3 (RGB) channels
            new_conv1.weight[:, :3, :, :] = original_conv1.weight.clone()
            # Initialize other channels by averaging the RGB weights
            new_conv1.weight[:, 3:, :, :] = torch.mean(
                original_conv1.weight, dim=1, keepdim=True
            ).repeat(1, 7, 1, 1)
        
        self.conv1 = new_conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        num_ftrs = resnet.fc.in_features
        self.fc = nn.Linear(num_ftrs, feature_vector_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Use gradient checkpointing to save memory
        x = checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer2, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer3, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.layer4, x, use_reentrant=False)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
