"""
Weather classification module.
Uses first two layers of pre-trained YOLOv3 object detection model.
"""

import argparse
from pathlib import Path
import logging
import opt
import torch
from lib import linear_classifier_head
from datetime import datetime

log = logging.getLogger('Main')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def make_dir(dir):
    if not dir.exists():
        dir.mkdir(parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Path to data root')
    parser.add_argument('--mode', default='predict', choices=['train', 'eval', 'predict'],
                        help='Running mode')
    parser.add_argument('--weights', default='checkpoints/CLS_WEATHER_head_weights.pt',
                        help='Weather classification weights (refers to the evaluation and test modes')
    parser.add_argument('--res_dir', default='results', help='Path to result directory')
    parser.add_argument('--novis', action='store_true',
                        help='Do not visualize weather classification results (refers to the evaluation and test modes)')

    args = parser.parse_args()
    return args


def main(opt, mode, data_root, res_dir, weights, novis=False):
    now = datetime.now()
    res_dir = Path(res_dir) / f'exp_{now.strftime("%Y%m%d_%H%M")}_{mode}'
    make_dir(res_dir)
    if not novis and mode != 'train':
        make_dir(res_dir / 'vis')
    log.addHandler(logging.FileHandler(res_dir / 'log.txt'))

    device = opt.device
    from DILAM.models import yolo
    model = yolo.Model('DILAM/models/yolov3.yaml', ch=3, nc=opt.obj_det_numcls).to(device)
    ckpt = torch.load(f'{Path(opt.ckpt_path) / opt.obj_det_clear_pretrained_model}', map_location=device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if mode == 'train':
        linear_classifier_head.train(model, opt, data_root, res_dir)
    if mode == 'eval':
        data_split = 'val'
        model.class_head.load_state_dict(torch.load(weights, map_location=device))
        linear_classifier_head.evaluate(model, opt, data_root, res_dir, data_split, novis,
                                        fname_weights=weights)
    if mode == 'predict':
        data_split = 'test'
        model.class_head.load_state_dict(torch.load(weights, map_location=device))
        linear_classifier_head.predict(model, opt, data_root, res_dir, data_split, novis)


if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        log.info('CUDA is available. Working on GPU')
        opt.device = torch.device('cuda')
    else:
        log.info('CUDA is not available. Working on CPU')
        opt.device = torch.device('cpu')

    log.info(f'Mode: {args.mode}')
    main(opt, args.mode, args.data_root, args.res_dir, args.weights, args.novis)
