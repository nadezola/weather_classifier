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
    parser.add_argument('--phase', default='test', help='Train or test phase')
    parser.add_argument('--res_dir', default='results', help='Path to result directory')
    parser.add_argument('--vis', action='store_true', help='Store weather classification visualizaion')

    args = parser.parse_args()
    return args


def main(opt, phase, data_root, res_dir):
    now = datetime.now()
    res_dir = Path(res_dir) / f'exp_{now.strftime("%Y%m%d_%H%M")}'
    make_dir(res_dir)

    log.addHandler(logging.FileHandler(res_dir / 'log.txt'))

    device = opt.device

    from DILAM.models import yolo
    model = yolo.Model('DILAM/models/yolov3.yaml', ch=3, nc=opt.obj_det_numcls).to(device)
    ckpt = torch.load(f'{Path(opt.ckpt_path) / opt.obj_det_clear_pretrained_model}', map_location=device)


    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if phase == 'train':
        linear_classifier_head.train_cls_head(model, opt, data_root, res_dir)



if __name__ == '__main__':
    args = parse_args()

    if torch.cuda.is_available():
        log.info('CUDA is available. Working on GPU')
        opt.device = torch.device('cuda')
    else:
        log.info('CUDA is not available. Working on CPU')
        opt.device = torch.device('cpu')

    log.info(f'Phase: {args.phase}')
    main(opt, args.phase, args.data_root, args.res_dir)
