"""
Weather classification module.
Uses first two layers of pre-trained YOLOv3 object detection model.

Usage:
    python main.py --data_root <path/to/data/root> --phase {'train'|'evaluate'} --res_dir <path/to/folder/where/save/results>
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
    parser.add_argument('--phase', default='test', choices=['train', 'eval', 'test'],
                        help='Running mode')
    parser.add_argument('--weights', default='checkpoints/CLS_WEATHER_head_weights.pt',
                        help='Weather classification weights (refers to the evaluation and test phases')
    parser.add_argument('--res_dir', default='results', help='Path to result directory')
    parser.add_argument('--novis', action='store_true',
                        help='Do not visualize weather classification results (refers to the evaluation and test phases)')

    args = parser.parse_args()
    return args


def main(opt, phase, data_root, res_dir, weights, novis=False):
    now = datetime.now()
    res_dir = Path(res_dir) / f'exp_{now.strftime("%Y%m%d_%H%M")}_{phase}'
    make_dir(res_dir)
    if not novis:
        make_dir(res_dir / 'vis')
    log.addHandler(logging.FileHandler(res_dir / 'log.txt'))

    device = opt.device
    from DILAM.models import yolo
    model = yolo.Model('DILAM/models/yolov3.yaml', ch=3, nc=opt.obj_det_numcls).to(device)
    ckpt = torch.load(f'{Path(opt.ckpt_path) / opt.obj_det_clear_pretrained_model}', map_location=device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if phase == 'train':
        linear_classifier_head.train(model, opt, data_root, res_dir)
    if phase == 'eval':
        model.class_head.load_state_dict(torch.load(weights, map_location=device))
        linear_classifier_head.evaluate(model, opt, data_root, res_dir, novis, fname_weights=weights)
    if phase == 'test':
        model.class_head.load_state_dict(torch.load(weights, map_location=device))
        linear_classifier_head.test(model, opt, data_root, res_dir, novis)


if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        log.info('CUDA is available. Working on GPU')
        opt.device = torch.device('cuda')
    else:
        log.info('CUDA is not available. Working on CPU')
        opt.device = torch.device('cpu')

    log.info(f'Phase: {args.phase}')
    main(opt, args.phase, args.data_root, args.res_dir, args.weights, args.novis)
