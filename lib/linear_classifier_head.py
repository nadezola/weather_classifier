import torch.nn as nn
import torch.optim as optim
import torch
from statistics import mean
from tqdm import tqdm
from lib.dataset import get_dataloader
import logging
from pathlib import Path
from sklearn import metrics
from datetime import datetime
import cv2
import matplotlib
import matplotlib.pyplot as plt


log = logging.getLogger('Main')


def validate(model, opt, dataset):
    device = opt.device
    model.eval()
    criterion = nn.CrossEntropyLoss()

    predictions = []
    gts = []
    loss_values = []
    for batch_idx, (imgs, labels, _) in (bar := tqdm(enumerate(dataset),
                                                  desc='\tValidation',
                                                  total=len(dataset),
                                                  unit=' batch')):
        with torch.no_grad():
            imgs, labels = imgs.to(device).float()/255, labels.to(device)
            outputs = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            targets = torch.zeros_like(outputs)
            for i, lbl in enumerate(labels):
                targets[i, lbl] = 1
            loss = criterion(outputs, targets.to(device))
            loss_values.append(loss.item())
            _, batch_pred = outputs.max(1)
            predictions.extend(batch_pred.tolist())
            gts.extend(labels.tolist())
            bar.set_postfix_str(f'batch_size {opt.batch_size}, loss {loss:.2f}')

    acc = metrics.accuracy_score(y_true=gts, y_pred=predictions)
    model.train()

    return acc, mean(loss_values)


def train(model, opt, data_root, res_dir,
                   fname_weights='CLS_WEATHER_head_weights.pt', fname_eval='train_evaluation.txt'):
    start_time = datetime.now()

    train_dataset = get_dataloader(opt, data_root, split='train', shuffle=True)
    val_dataset = get_dataloader(opt, data_root, split='val', shuffle=True)

    device = opt.device
    optimizer = optim.SGD(model.class_head.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Train
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
        for batch_idx, (imgs, labels, _) in (bar := tqdm(enumerate(train_dataset),
                                                                    desc=f'\tTrain',
                                                                    total=len(train_dataset),
                                                                    unit=' batch')):
            optimizer.zero_grad()
            imgs = imgs.to(device).float() / 255.0
            out = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            targets = torch.zeros_like(out)
            for i, lbl in enumerate(labels):
                targets[i, lbl] = 1
            loss = criterion(out, targets.to(device))
            loss.backward()
            optimizer.step()
            bar.set_postfix_str(f'batch_size {opt.batch_size}, loss {loss:.2f}, lr {scheduler.get_last_lr()[0]}')
        scheduler.step()

        # Validation
        acc, loss = validate(model, opt, val_dataset)
        log.info(f'\t:: Acc={acc * 100:.1f} :: Loss={loss:.2f}')

        # Save best
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_epoch = epoch
            torch.save(model.class_head.state_dict(), Path(res_dir) / fname_weights)
            log.info(f'>> New best saved to {Path(res_dir) / fname_weights}\n')

    runtime = datetime.now() - start_time

    # Report
    log.info(f'Done.\nTrained {epoch} epochs in a time of {runtime}.\nBest Loss = {best_loss:.2f} on Epoch = {best_epoch} '
             f'\nAcc = {best_acc * 100}.')
    with open(Path(res_dir) / fname_eval, 'w') as f:
        print(f'Mode Train\n', file=f)
        print(f'Epochs = {opt.epochs}', file=f)
        print(f'Batch size = {opt.batch_size}', file=f)
        print(f'Image size = {opt.img_size}', file=f)
        print(f'Classes = {opt.CLS_WEATHER}', file=f)
        print(f'Pretrained Detection Model = {opt.obj_det_clear_pretrained_model}\n'
              f'  for {opt.obj_det_numcls} Object Classes = {opt.obj_det_cls}', file=f)
        print(f'Augment = {opt.augment}', file=f)
        print(f'Number workers = {opt.workers}\n', file=f)
        print(f'Running Time = {runtime}', file=f)
        print(f'Best Loss = {best_loss:.2f} on Epoch = {best_epoch}\nAcc = {best_acc * 100}', file=f)


def evaluate(model, opt, data_root, res_dir, data_split, novis, fname_weights='', fname_eval='res_evaluation.txt'):
    device = opt.device
    model.eval()
    criterion = nn.CrossEntropyLoss()

    eval_dataset = get_dataloader(opt, data_root, split=data_split, shuffle=False)

    # Run
    predictions = []
    gts = []
    loss_values = []
    for batch_idx, (imgs, labels, img_names) in (bar := tqdm(enumerate(eval_dataset),
                                                             desc=f'Evaluation',
                                                             total=len(eval_dataset),
                                                             unit=' batch')):
        with torch.no_grad():
            imgs, labels = imgs.to(device).float() / 255, labels.to(device)
            outputs = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            targets = torch.zeros_like(outputs)
            for i, lbl in enumerate(labels):
                targets[i, lbl] = 1
            loss = criterion(outputs, targets.to(device))
            loss_values.append(loss.item())
            _, batch_preds = outputs.max(1)
            predictions.extend(batch_preds.tolist())
            gts.extend(labels.tolist())

            if not novis:
                for img_name, pred, lbl in zip(img_names, batch_preds.tolist(), labels.tolist()):
                    s = f'Prediction - {opt.CLS_WEATHER[pred]} :: Label - {opt.CLS_WEATHER[lbl]}'
                    img = cv2.imread(str(Path(data_root) / 'images' / data_split / img_name))
                    img = cv2.putText(img, text=s, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                      color=(0, 0, 0), thickness=2)
                    cv2.imwrite(str(res_dir / 'vis' / img_name), img)

            bar.set_postfix_str(f'batch_size {opt.batch_size}, loss {loss:.2f}')

    # Metrics
    acc = metrics.accuracy_score(y_true=gts, y_pred=predictions)
    loss = mean(loss_values)
    conf_matrix = metrics.confusion_matrix(y_true=gts, y_pred=predictions, normalize='pred')
    # res_eval_dic = metrics.classification_report(y_true=gts, y_pred=predictions, target_names=opt.CLS_WEATHER,
    #                                            output_dict=True, zero_division=0.)
    res_eval = metrics.classification_report(y_true=gts, y_pred=predictions, target_names=opt.CLS_WEATHER,
                                               output_dict=False, zero_division=0.)

    # Report
    log.info(f':: Acc={acc * 100:.1f} :: Loss={loss:.2f}')
    log.info(res_eval)
    log.info(f'Confusion Matrix:\n{conf_matrix}')
    with open(Path(res_dir) / fname_eval, 'w') as f:
        print(f'Mode Evaluation\n', file=f)
        print(f'Classes = {opt.CLS_WEATHER}', file=f)
        print(f'Weights for the Classification Head = {fname_weights}', file=f)
        print(f'Pretrained Detection Model = {opt.obj_det_clear_pretrained_model}\n'
              f'  for {opt.obj_det_numcls} Object Classes = {opt.obj_det_cls}', file=f)
        print(f'Image size = {opt.img_size}', file=f)
        print(f'Augment = {opt.augment}\n', file=f)
        print(f'Loss = {loss:.2f}\n', file=f)
        print(res_eval, file=f)
        print(f'Confusion Matrix:\n{conf_matrix}', file=f)

    # Plot Confusion matrix
    metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=opt.CLS_WEATHER).plot(cmap='Blues')
    plt.savefig(Path(res_dir) / 'confusion_matrix.png')

def demo(model, opt, data_root, res_dir, data_split, novis):
    device = opt.device
    model.eval()

    test_dataset = get_dataloader(opt, data_root, split=data_split, phase='test', shuffle=False)

    # Run
    for batch_idx, (imgs, _, img_names) in tqdm(enumerate(test_dataset), desc=f'Test', total=len(test_dataset), unit=' batch'):
        with torch.no_grad():
            imgs = imgs.to(device).float()/255
            outputs = model.forward_cls(imgs).view([-1, len(opt.CLS_WEATHER)])
            _, batch_preds = outputs.max(1)
            if not novis:
                for img_name, pred in zip(img_names, batch_preds.tolist()):
                    s = f'Prediction - {opt.CLS_WEATHER[pred]} '
                    img = cv2.imread(str(Path(data_root) / 'images' / 'test' / img_name))
                    img = cv2.putText(img, text=s, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                      color=(0, 0, 0), thickness=2)
                    cv2.imwrite(str(res_dir / 'vis' / img_name), img)
    log.info(f'Done')
