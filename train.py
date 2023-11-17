# -*- coding: utf-8 -*-
# @Author   : zqian9
import os
import time
import random
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import config
from dataset import ASVSpoof19
from model import resnetv2_18, resnetv2_34, resnetv2_10

# set used GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def validate(dataloader, model, loss_func, device):
    running_loss = []
    num_correct = 0
    num_smamples = 0
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y, _, _ in dataloader:
            num_smamples += batch_x.size(0)

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_out = model(batch_x)

            batch_loss = loss_func(batch_out, batch_y)
            batch_pred = torch.argmax(F.softmax(batch_out, dim=1), dim=1)
            num_correct += batch_pred.eq(batch_y).sum(dim=0).item()
            running_loss.append(batch_loss.item())

    return num_correct / num_smamples, np.mean(running_loss)


def train(train_dataloader, valid_dataloader,
          model, num_epochs, initial_lr, ckpt_root, device):
    best_val_acc = 0.0

    loss_weight = torch.FloatTensor([0.9, 0.1]).to(device)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00005)

    for epoch in range(num_epochs):
        start_time = time.time()

        running_loss = []
        num_correct = 0
        num_smamples = 0
        model.train()

        for i, (batch_x, batch_y, _, _) in enumerate(train_dataloader):
            num_smamples += batch_x.size(0)

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            batch_out = model(batch_x)
            batch_loss = loss_func(batch_out, batch_y)
            batch_pred = torch.argmax(F.softmax(batch_out, dim=1), dim=1)
            num_correct += batch_pred.eq(batch_y).sum(dim=0).item()

            batch_loss.backward()
            optimizer.step()
            running_loss.append(batch_loss.item())

            if i % 10 == 0:
                print('\rIteration: {:4d} loss: {:6.4f} - acc: {:5.2f}%'.format(
                    i, batch_loss.item(),
                    batch_pred.eq(batch_y).sum(dim=0).item() / batch_x.size(0) * 100), end="")

        train_loss = np.mean(running_loss)
        train_acc = num_correct / num_smamples
        scheduler.step()

        val_acc, val_loss = validate(valid_dataloader, model, loss_func, device)

        end_time = time.time()
        cost_time = int(end_time - start_time)

        out_info = ("\rEpoch:{:3d}/{:3d} - duration:{:4d}sec - train loss:{:6.4f} "
                    "- train acc:{:5.2f}% - val loss:{:6.4f} - val acc:{:5.2f}%").format(
            epoch + 1, num_epochs, cost_time,
            train_loss, train_acc * 100, val_loss, val_acc * 100)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            out_info += " - Current Best ACC Model"

            torch.save(model.state_dict(), ckpt_root / "best_acc_model.pt")

        print(out_info)


def main():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] Training on device: {}'.format(device.upper()))

    model = resnetv2_10().to(device)
    print('[INFO] Model loaded...')

    # define model save directory
    ckpt_root = Path(config.ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # load data
    train_dataset = ASVSpoof19(
        data_root=config.data_root,
        duration=config.duration,
        dataset_type='train'
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    valid_dataset = ASVSpoof19(
        data_root=config.data_root,
        duration=config.duration,
        dataset_type='dev'
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    train(train_dataloader, valid_dataloader,
          model, config.epochs, config.init_lr, ckpt_root, device)


if __name__ == '__main__':
    main()
