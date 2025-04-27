import os
from Train import SingleLabelResNet
from Tester import get_test_loader, run_test
from Dataloader import split_dataset
import Config

# 💾 Paths
BASE_DIR = BSE
IMG_DIR = IMG
MODEL_PATH = MDL

TRAIN_CSV = TRN
VAL_CSV = VAL
TEST_CSV = TST

def train_model():
    print("🚀 Starting Training...")
    model = SingleLabelResNet(
        csv_path=TRAIN_CSV,
        img_dir=IMG_DIR,
        epochs=epchs
    )
    model.train()

def test_model():
    print("\n🧪 Starting Testing...")
    test_loader, class_names = get_test_loader(TEST_CSV, IMG_DIR)
    run_test(MODEL_PATH, test_loader, class_names)


if __name__ == "__main__":
    # ✅ Flags
    RUN_TRAINING = R_TRN
    RUN_TESTING = R_TST

    if RUN_TRAINING:
        train_model()

    if RUN_TESTING:
        test_model()
