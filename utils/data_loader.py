# utils/data_loader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter


class MURADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Custom dataset for MURA XR_ELBOW images.
        Args:
            root_dir (str): Path to MURA-v1.1 root folder.
            split (str): 'train' or 'valid'
            transform: torchvision transforms
        """
        self.samples = []
        self.transform = transform

        # Only use XR_ELBOW folder
        split_dir = os.path.join(root_dir, split, 'XR_ELBOW')

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"❌ Path not found: {split_dir}")

        for patient_folder in os.listdir(split_dir):
            patient_path = os.path.join(split_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            for study_folder in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_folder)

                # Correct label assignment based on study folder name
                label = 1 if 'positive' in study_folder.lower() else 0

                # Recursively find .png images in subfolders
                for root, _, files in os.walk(study_path):
                    for file in files:
                        if file.endswith('.png'):
                            img_path = os.path.join(root, file)
                            self.samples.append((img_path, label))

        # 🔍 Print class distribution
        labels = [label for _, label in self.samples]
        label_counts = Counter(labels)
        print(f"[{split.upper()}] Loaded samples - Class 0 (Negative): {label_counts[0]}, Class 1 (Positive): {label_counts[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(root_dir="data/MURA-v1.1", batch_size=32, img_size=224):
    """
    Creates PyTorch dataloaders for XR_ELBOW classification.

    Returns:
        dict: { 'train': DataLoader, 'valid': DataLoader }
    """

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = MURADataset(root_dir=root_dir, split='train', transform=train_transform)
    valid_dataset = MURADataset(root_dir=root_dir, split='valid', transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return {'train': train_loader, 'valid': valid_loader}
