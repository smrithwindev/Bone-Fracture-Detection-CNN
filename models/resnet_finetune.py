# models/resnet_finetune.py

import torch.nn as nn
from torchvision import models

def get_finetuned_resnet18(num_classes=2):
    """
    Loads a pretrained ResNet18 and replaces the classifier for fine-tuning.

    Args:
        num_classes (int): Number of output classes (default: 2 for fracture/no fracture)

    Returns:
        model: Modified ResNet18 model
    """
    model = models.resnet18(pretrained=True)

    # 🔒 Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # 🔓 Unfreeze deeper layers for fine-tuning
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 🔓 Replace and unfreeze the final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
