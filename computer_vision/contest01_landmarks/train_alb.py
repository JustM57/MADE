"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from augmentations import test_transforms, image_augmentations, key_points_augmentations

from utils import NUM_PTS
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
scaler = GradScaler()


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"].to(device)  # B x (2 * NUM_PTS)
        with autocast():
            pred_landmarks = model(images)  # B x (2 * NUM_PTS)
            loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.append(loss.item())
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss, real_val_loss = [], []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

        # Расчет "правильного" лосса
        fs = batch["scale_coef"].numpy()
        # Вытаскиваем инфо о кромках
        margins_x = batch["crop_margin_x"].numpy()
        margins_y = batch["crop_margin_y"].numpy()
        # Пересчитываем в исходные координаты предсказания модели
        pred_landmarks = pred_landmarks.cpu().numpy().reshape((len(pred_landmarks), NUM_PTS, 2))
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)
        # Пересчитываем в исходные координаты ground_true - координаты
        landmarks = landmarks.cpu().numpy().reshape((len(pred_landmarks), NUM_PTS, 2))
        real_landmarks = restore_landmarks_batch(landmarks, fs, margins_x, margins_y)
        # Добавяем MSE в список real_val_loss
        real_loss = (prediction.reshape(-1) - real_landmarks.reshape(-1)) ** 2
        real_val_loss.append(np.mean(real_loss))

    return np.mean(val_loss), np.mean(real_val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, "train"), test_transforms, image_augmentations=image_augmentations,
        key_points_augmentations=key_points_augmentations, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=11, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), test_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=11, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    model = models.regnet_y_8gf(pretrained=True)
    model.requires_grad_(False)
    # model.classifier = nn.Sequential(
    #     nn.LayerNorm([768, 1, 1], eps=1e-06, elementwise_affine=True),
    #     nn.Flatten(start_dim=1, end_dim=-1),
    #     nn.Linear(in_features=768, out_features=2 * NUM_PTS, bias=True)
    # )
    # model.classifier.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer1 = optim.Adam(model.parameters(), lr=0.1)
    # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='min',
    #     factor=0.5,
    #     cooldown=5,
    #     patience=3,
    #     verbose=True
    # )
    scheduler1 = optim.lr_scheduler.StepLR(
        optimizer1, step_size=len(train_dataloader) // 6, gamma=0.5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_dataloader),
        T_mult=2,
        eta_min=1e-6,
        last_epoch=-1,
    )

    loss_fn = fnn.l1_loss

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    train_loss = train(model, train_dataloader, loss_fn, optimizer1, scheduler=scheduler1, device=device)
    val_loss, real_val_loss = validate(model, val_dataloader, loss_fn, device=device)
    print(f"Epoch #-1:\ttrain loss: {round(train_loss, 3)}\tval loss: {round(val_loss, 3)}\treal val loss: {round(real_val_loss, 3)}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
            torch.save(model.state_dict(), fp)
    model.requires_grad_(True)

    # 2. train & validate
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, scheduler=scheduler, device=device)
        val_loss, real_val_loss = validate(model, val_dataloader, loss_fn, device=device)
        print(f"Epoch #{epoch}:\ttrain loss: {round(train_loss, 3)}\tval loss: {round(val_loss, 3)}\treal val loss: {round(real_val_loss, 3)}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), test_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=11, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
