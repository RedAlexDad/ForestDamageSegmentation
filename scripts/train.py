"""Training script for U-Net model."""
import os
import argparse
import numpy as np

from src.data import load_train_data, create_datasets, get_training_augmentor
from src.models import build_unet, compile_model
from src.losses import dice_loss, combined_loss
from src.metrics import dice_coefficient, iou_score
from src.training import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net model")
    parser.add_argument("--data-dir", type=str, default="data/train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-channels", type=int, default=6)
    parser.add_argument("--model-path", type=str, default="models/unet_model.h5")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    images, masks = load_train_data(args.data_dir, args.n_channels)
    print(f"Loaded {len(images)} images")

    print("Creating datasets...")
    train_ds, val_ds = create_datasets(
        images, masks,
        batch_size=args.batch_size,
        val_split=0.2
    )

    print("Building model...")
    model = build_unet(input_size=(256, 256, args.n_channels))
    model = compile_model(
        model,
        learning_rate=args.lr,
        loss=dice_loss,
        metrics=[dice_coefficient, iou_score]
    )
    model.summary()

    print("Training...")
    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "n_channels": args.n_channels,
        "image_size": 256
    }

    trainer = ModelTrainer(model, config)
    history = trainer.train(
        train_ds,
        val_ds,
        epochs=args.epochs,
        verbose=1
    )

    os.makedirs(os.path.dirname(args.model_path) or "models", exist_ok=True)
    trainer.save(args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()