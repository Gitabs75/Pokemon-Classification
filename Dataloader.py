import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from sklearn.model_selection import train_test_split
imprt Config

class PokemonDataset(Dataset):
    def __init__(self, csv_path, img_dir, class_names, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_names.index(row['type'])
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(csv_path, img_dir, class_names, image_size=Config.img_size, batch_size=Config.btch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    df = pd.read_csv(csv_path)
    labels = df['type'].apply(lambda x: class_names.index(x)).values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=Config.t_size, random_state=42)
    for train_idx, val_idx in splitter.split(df, labels):
        train_data = df.iloc[train_idx].reset_index(drop=True)
        val_data = df.iloc[val_idx].reset_index(drop=True)

    full_dataset = PokemonDataset(csv_path, img_dir, class_names, transform)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def split_dataset(base_dir, csv_filename=lbl_csv, output_folder=otp_flder):
    csv_path = os.path.join(base_dir, csv_filename)
    output_dir = os.path.join(base_dir, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract Pokémon name as group identifier
    df["pokemon"] = df["filename"].apply(lambda x: x.split("_")[0])

    # Group by Pokémon name and keep only one row per Pokémon (for stratification)
    grouped = df.groupby("pokemon").first().reset_index()

    # Stratified split based on label (type) using grouped Pokémon
    train_val_pokemon, test_pokemon = train_test_split(
        grouped,
        test_size=0.15,
        stratify=grouped["type"],
        random_state=42
    )

    train_pokemon, val_pokemon = train_test_split(
        train_val_pokemon,
        test_size=Config.tst_sze,  # 0.1765 * 0.85 ≈ 0.15 total for val
        stratify=train_val_pokemon["type"],
        random_state=42
    )

    # Get actual splits using Pokémon name groupings
    train_df = df[df["pokemon"].isin(train_pokemon["pokemon"])]
    val_df   = df[df["pokemon"].isin(val_pokemon["pokemon"])]
    test_df  = df[df["pokemon"].isin(test_pokemon["pokemon"])]

    # Drop helper column
    train_df = train_df.drop(columns=["pokemon"])
    val_df   = val_df.drop(columns=["pokemon"])
    test_df  = test_df.drop(columns=["pokemon"])

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Stratified dataset split complete:")
    print(f"  ➤ Train: {len(train_df)} samples")
    print(f"  ➤ Val:   {len(val_df)} samples")
    print(f"  ➤ Test:  {len(test_df)} samples")

    return train_path, val_path, test_path