import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from lane_dataset import LaneDataset
from unet1 import UNet
from mixed_loss import mixed_loss
from tqdm import tqdm
import os
from torch.amp import autocast
import matplotlib.pyplot as plt
from iou import compute_iou


# 경로 설정
TRAIN_LIST = "lane_detection/cnn/SDLane/train/train_list.txt"
TRAIN_IMAGES = "lane_detection/cnn/SDLane/train/resized_images"
TRAIN_MASKS = "lane_detection/cnn/SDLane/train/resized_masks"
SAVE_PATH = "best_model.pth"

# 전체 dataset 생성
dataset = LaneDataset(TRAIN_LIST, TRAIN_IMAGES, TRAIN_MASKS)

# 비율로 나누기 (예: train 80%, val 20%)
val_ratio = 0.2
val_len = int(len(dataset) * val_ratio)
train_len = len(dataset) - val_len

# 고정된 시드로 재현성 유지
train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

# 각 DataLoader 생성
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

# 모델 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=3).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# AMP (자동 mixed precision) 설정 (선택)
scaler = torch.amp.GradScaler(device="cuda")

print(f"Train size: {len(train_set)}, Validation size: {len(val_set)}")


# 학습 루프
num_epochs = 5
best_val_loss = float("inf")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    print(f"[Epoch : {epoch+1}] Training...")
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # loss = mixed_loss(outputs, masks)
        loss = F.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0.0
    iou_scores = []

    with torch.no_grad():
        print(f"[Epoch : {epoch+1}] Validating...")
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # loss = mixed_loss(outputs, masks)
            loss = F.cross_entropy(outputs, masks)
            val_loss += loss.item()

            # === IoU 계산 ===
            batch_ious = compute_iou(outputs, masks, num_classes=3)  # 클래스 수 맞게 설정
            iou_scores.append(batch_ious)

    val_loss /= len(val_loader)
    avg_iou = torch.tensor(iou_scores).nanmean(dim=0)  # class별 평균 IoU
    mean_iou = avg_iou.nanmean().item()  # 전체 클래스 평균
    print(f"Mean IoU: {mean_iou}")

    # === 모델 저장 ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model improved and saved at epoch {epoch+1}!")

    # === 로그 출력 ===
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

print("학습 완료!")

# === 시각화 ===

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.savefig("./loss_result/loss_plot.png")
plt.show()