import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets_onehot, dim=(0, 2, 3))
        union = torch.sum(preds + targets_onehot, dim=(0, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()  # 클래스 평균
        return loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # 기본 CE
        pt = torch.exp(-ce_loss)  # pt = softmax 결과의 정답 클래스 확률

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at = self.alpha[targets]
            focal_loss = focal_loss * at

        return focal_loss.mean() if self.reduction == "mean" else focal_loss

# 통합 Loss 함수
def mixed_loss(preds, targets, alpha=0.5, gamma=2.0):
    dice = DiceLoss()(preds, targets)
    focal = FocalLoss(gamma=gamma)(preds, targets)
    return alpha * dice + (1 - alpha) * focal