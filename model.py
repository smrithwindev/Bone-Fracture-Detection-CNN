# model.py

from models.resnet_finetune import get_finetuned_resnet18

def build_model(num_classes=2):
    """
    Builds and returns the fine-tuned ResNet18 model.
    """
    model = get_finetuned_resnet18(num_classes=num_classes)
    return model
