import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from Model import CustomResNet18
import Config

# 🔥 predict function without any classes
def predict(model_path, image_folder, save_csv=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = Config.CLS
    batch_size = Config.pbatch

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Collect and process images
    image_paths = [os.path.join(image_folder, fname)
                   for fname in os.listdir(image_folder)
                   if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("❌ No images found in the folder.")
        return

    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")

    images_tensor = torch.stack(images)
    dataset = TensorDataset(images_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = CustomResNet18(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []

    img_idx = 0  # manual index to match images and paths
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for pred in preds:
                img_path = os.path.basename(valid_paths[img_idx])
                predictions.append((img_path, class_names[pred]))
                img_idx += 1

    # Save or return results
    results_df = pd.DataFrame(predictions, columns=["Filename", "Predicted Class"])
    if save_csv:
        save_path = os.path.join(image_folder, "predictions.csv")
        results_df.to_csv(save_path, index=False)
        print(f"✅ Predictions saved to {save_path}")

    return results_df
