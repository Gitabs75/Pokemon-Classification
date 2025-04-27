import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Model import CustomResNet18

# --- 🧠 Class Names (must match training order) ---
CLASS_NAMES = [
    'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting',
    'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
    'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
]

# --- ⚙️ Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 🧼 Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# --- 📂 Dataset ---
class PokemonTestDataset(Dataset):
    def __init__(self, csv_path, img_dir, class_names, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names
        self.name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples = [
            (row["filename"], self.name_to_idx[row["type"]])
            for _, row in self.data.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 🔁 Loader Function ---
def get_test_loader(csv_path, img_dir, batch_size=32):
    dataset = PokemonTestDataset(csv_path, img_dir, CLASS_NAMES, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, CLASS_NAMES

# --- 🧪 Test Function (can be used in scripts or standalone) ---
def run_test(model_path, test_loader, class_names, save_path="conf_matrix_test.png"):
    model = CustomResNet18(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("📋 Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Test confusion matrix saved as {save_path}")


