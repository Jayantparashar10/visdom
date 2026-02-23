#!/usr/bin/env python3

# Copyright 2017-present, The Visdom Authors
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Trains a small CNN on MNIST and logs training progress to Visdom.
# Each metric is logged with an explicit viz.line() call to show
# what manual instrumentation looks like in practice.
#
# Dashboard windows: loss | accuracy | lr (StepLR) | grad_norm | samples
#
# Usage:
#   python -m visdom.server          # start server in one terminal
#   python example/pytorch_mnist_demo.py   # run this in another
#   open http://localhost:8097

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import visdom

# config #

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset #

transform = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ]
)

# model #


# small LeNet-style CNN: 2 conv layers + 2 fc layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# training #


# sum of squared gradient norms across all parameters
def grad_norm(mdl):
    total = 0.0
    for p in mdl.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total**0.5


def train_epoch(mdl, loader, optimizer, criterion):
    mdl.train()
    running_loss = 0.0
    norm = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(mdl(images), labels)
        loss.backward()
        norm = grad_norm(mdl)  # capture norm before the weights move
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader), norm


def evaluate(mdl, loader):
    mdl.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (mdl(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":

    # dataset #

    train_dataset = tv.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = tv.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # visdom #

    viz = visdom.Visdom(port=8097)

    assert (
        viz.check_connection()
    ), "No connection could be formed quickly, start with: python -m visdom.server"

    # show a sample of images before training starts
    # de-normalise from [-1, 1] back to [0, 1] for display
    sample_images, _ = next(iter(train_loader))
    viz.images(
        sample_images[:8] * 0.5 + 0.5,
        nrow=4,
        padding=6,
        win="samples",
        opts=dict(title="MNIST Samples", width=400, height=120),
    )

    print(f"\ntraining on {DEVICE}  |  epochs={EPOCHS}  lr={LR}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, norm = train_epoch(model, train_loader, optimizer, criterion)
        accuracy = evaluate(model, test_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # each metric needs its own viz.line call — this is the repetition
        # that a future VisdomLogger will collapse into a single log() call
        # update=None on epoch 1 wipes any stale window from a previous run,
        # then 'append' adds one point per epoch so the chart grows live
        upd = None if epoch == 1 else "append"

        # training loss
        viz.line(
            Y=np.array([train_loss]),
            X=np.array([epoch]),
            win="loss",
            update=upd,
            opts=dict(title="Training Loss", xlabel="Epoch", ylabel="Loss"),
        )

        # validation accuracy
        viz.line(
            Y=np.array([accuracy]),
            X=np.array([epoch]),
            win="accuracy",
            update=upd,
            opts=dict(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)"),
        )

        # learning rate
        viz.line(
            Y=np.array([current_lr]),
            X=np.array([epoch]),
            win="lr",
            update=upd,
            opts=dict(title="Learning Rate", xlabel="Epoch", ylabel="LR"),
        )

        # gradient norm — useful for spotting vanishing/exploding gradients
        viz.line(
            Y=np.array([norm]),
            X=np.array([epoch]),
            win="grad_norm",
            update=upd,
            opts=dict(title="Gradient Norm", xlabel="Epoch", ylabel="||grad||"),
        )

        print(
            f"epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  acc={accuracy:.2f}%  "
            f"lr={current_lr:.5f}  grad_norm={norm:.4f}"
        )

        scheduler.step()

    print("\ndone — open http://localhost:8097 to view the dashboard")
