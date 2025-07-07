
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import time
import copy

from models.resnet_finetune import get_finetuned_resnet18
from utils.data_loader import create_dataloaders

def train_model(model, dataloaders, device, num_epochs=30, learning_rate=1e-5, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}")

            if phase == 'valid':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "models/best_model.pth")
                    print("✅ Saved new best model.")
                    counter = 0  # Reset counter when improvement happens
                else:
                    counter += 1
                    print(f"⏸️ No improvement in F1. EarlyStopping counter: {counter}/{patience}")
                    if counter >= patience:
                        print("🛑 Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return model

    print("\n🎉 Training complete. Best Validation F1: {:.4f}".format(best_f1))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading model...")
    model = get_finetuned_resnet18()
    model.to(device)

    print("📂 Preparing dataloaders...")
    dataloaders = create_dataloaders(root_dir="D:/CNN_projects/bone_fracture_detection/data/MURA-v1.1")

    print("🚀 Starting training...")
    trained_model = train_model(model, dataloaders, device, num_epochs=30, patience=7)

    print("✅ Training finished. Best model saved to models/best_model.pth")

