"""재밍 분류 CNN 정의 + 학습.

사용:
    python train_cnn.py                       # dataset.npz로 학습, cnn_best.pt 저장
    python train_cnn.py --epochs 30 --lr 5e-4
다른 스크립트에서는 load_cnn, predict_proba를 import해서 사용한다.
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class JammingCNN(nn.Module):
    """입력 [B,1,32,32] 스펙트로그램 -> 클래스 로짓. 클래스 0은 항상 'none'(정상)."""

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 4x4
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn(path, device=None):
    """저장된 체크포인트 -> (eval 모드 모델, 클래스 이름 리스트)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    classes = list(ckpt["classes"])
    model = JammingCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


@torch.no_grad()
def predict_proba(model, x, device=None, batch=512):
    """numpy 이미지 배열 [N,1,32,32] -> softmax 확률 [N,C]."""
    device = device or next(model.parameters()).device
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        outs.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cnn_best.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    d = np.load(args.data)
    classes = [str(c) for c in d["classes"]]
    assert classes[0] == "none", "클래스 0은 'none'이어야 합니다"

    def loader(x, y, shuffle):
        ds = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    train_loader = loader(d["x_train"], d["y_train"], True)
    val_loader = loader(d["x_val"], d["y_val"], False)
    test_loader = loader(d["x_test"], d["y_test"], False)

    model = JammingCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None
    print(f"device={device} classes={classes} train={len(d['y_train'])} val={len(d['y_val'])}")
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, device, opt)
        vl, va = run_epoch(model, val_loader, device)
        sched.step()
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:02d}/{args.epochs} | train loss {tl:.4f} acc {ta:.3f} | "
              f"val loss {vl:.4f} acc {va:.3f}{' *' if va == best_val else ''}")

    model.load_state_dict(best_state)
    _, test_acc = run_epoch(model, test_loader, device)
    print(f"best val acc {best_val:.4f} | test acc {test_acc:.4f}")

    torch.save({"state_dict": best_state, "classes": classes}, args.out)
    with open("train_history.json", "w") as f:
        json.dump({**hist, "best_val_acc": best_val, "test_acc": test_acc}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ep_axis = np.arange(1, args.epochs + 1)
    a1.plot(ep_axis, hist["train_loss"], label="train")
    a1.plot(ep_axis, hist["val_loss"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Loss"); a1.legend(); a1.grid(ls=":")
    a2.plot(ep_axis, hist["train_acc"], label="train")
    a2.plot(ep_axis, hist["val_acc"], label="val")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy"); a2.set_title("Accuracy"); a2.legend(); a2.grid(ls=":")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    print(f"saved {args.out}, train_history.json, training_curves.png")


if __name__ == "__main__":
    main()
