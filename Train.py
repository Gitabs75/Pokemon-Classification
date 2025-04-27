import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from Model import CustomResNet18
from Dataloader import get_dataloaders
import Config

def train(csv_path, img_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = sorted(Config.CLS)
    image_size = Config.img_sze
    batch_size = Config.btch_sze
    epochs = Config.epchs
    log_file = os.path.join(Config.BSE, Config.TRN_LOG)
    model_path = Config.MDL

    train_loader, val_loader = get_dataloaders(
        csv_path, img_dir, class_names, image_size, batch_size
    )

    model = CustomResNet18(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.l, weight_decay=Config.wd)

    logs = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, val_report = evaluate(model, val_loader, criterion, class_names, device, epoch)

        logs.append([epoch + 1, avg_loss, val_loss])
        print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
        print(val_report)
        torch.save(model.state_dict(), model_path)

    pd.DataFrame(logs, columns=['Epoch', 'Train Loss', 'Val Loss']).to_csv(log_file, index=False)
    print(f"✅ Training Complete. Model saved to {model_path}")

def evaluate(model, val_loader, criterion, class_names, device, epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_epoch_{epoch + 1}.png")
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    return total_loss / len(val_loader), report
