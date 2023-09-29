import torch.nn as nn
import torch.optim as optim
import torch
from statistics import mean
from tqdm import tqdm
from lib.dataset import get_dataloader
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
from datetime import datetime


log = logging.getLogger('Main')

def validate_cls_head(model, opt, val_dataset):
    device = opt.device
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_gts = []
    loss_values = []
    for batch_idx, (imgs, labels) in (bar := tqdm(enumerate(val_dataset),
                                                  desc=f'\tValidate',
                                                  total=len(val_dataset),
                                                  unit=' batch')):
        with torch.no_grad():
            imgs, labels = imgs.to(device).float()/255, labels.to(device)
            outputs = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            gts = torch.zeros_like(outputs)
            for i, lbl in enumerate(labels):
                gts[i, lbl] = 1
            loss = criterion(outputs, gts.to(device))
            loss_values.append(loss.item())
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.tolist())
            all_gts.extend(labels.tolist())
            bar.set_postfix_str(f'batch_size {opt.batch_size}, loss {loss:.2f}')

    acc = accuracy_score(y_true=all_gts, y_pred=all_predictions)
    model.train()

    return acc, mean(loss_values)


def train_cls_head(model, opt, data_root, res_dir, 
                   fname_weights='CLS_WEATHER_head_weights.pt', fname_eval='train_evaluation.txt'):
    start_time = datetime.now()
    device = opt.device

    train_dataset = get_dataloader(opt, data_root, 'train', shuffle=True)
    val_dataset = get_dataloader(opt, data_root, 'val', shuffle=True)

    optimizer = optim.SGD(model.class_head.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_loss = 10e+10
    best_acc = 0
    best_epoch = -1
    for epoch in range(1, opt.epochs + 1):
        for m in model.modules():
            m.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                m.eval()

        for m in model.class_head.modules():
            m.requires_grad_(True)

        for m in model.class_head.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm3d):
                m.train()

        log.info(f'Epoch {epoch}/{opt.epochs}')
        for batch_idx, (imgs, labels) in (bar := tqdm(enumerate(train_dataset),
                                                                    desc=f'\tTrain',
                                                                    total=len(train_dataset),
                                                                    unit=' batch')):
            optimizer.zero_grad()
            imgs = imgs.to(device).float() / 255.0
            out = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            gts = torch.zeros_like(out)
            for i, lbl in enumerate(labels):
                gts[i, lbl] = 1
            loss = criterion(out, gts.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            bar.set_postfix_str(f'batch_size {opt.batch_size}, loss {loss:.2f}')

        # Validation
        acc, loss = validate_cls_head(model, opt, val_dataset)
        log.info(f'\t:: Acc={acc * 100:.1f} :: Loss={loss:.2f}')

        # Save
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_epoch = epoch
            torch.save(model.class_head.state_dict(), Path(res_dir) / fname_weights)
            log.info(f'>> New best saved to {Path(res_dir) / fname_weights}\n')

    runtime = datetime.now() - start_time

    log.info(f'Done.\nTrained {epoch} epochs in a time of {runtime}.\nBest Loss = {best_loss:.2f} on Epoch = {best_epoch} '
             f'\nAcc = {best_acc * 100}.')

    with open(Path(res_dir) / fname_eval, 'w') as f:
        print(f'Phase "train"\n', file=f)
        print(f'Epochs = {opt.epochs}', file=f)
        print(f'Batch_size = {opt.batch_size}', file=f)
        print(f'Image_size = {opt.img_size}', file=f)
        print(f'CLS_WEATHER = {opt.CLS_WEATHER}', file=f)
        print(f'Pretrained_det_model: {opt.obj_det_clear_pretrained_model} for {opt.obj_det_numcls}', file=f)
        print(f'Augment = {opt.augment}', file=f)
        print(f'Number_workers = {opt.workers}\n\n', file=f)
        print(f'Time: {runtime}', file=f)
        print(f'Best Loss: {best_loss:.2f} on Epoch {best_epoch}\nAcc: {best_acc * 100}', file=f)

