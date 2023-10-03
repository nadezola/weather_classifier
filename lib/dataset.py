from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class ImageDataset(Dataset):
    def __init__(self, opt, data_root, split, phase):
        self.img_size = opt.img_size
        self.CLS_WEATHER = opt.CLS_WEATHER
        self.augment = opt.augment
        self.phase = phase
        self.img_files = []
        self.labels = []

        if self.phase == 'test':
            p = Path(data_root) / 'images' / split
            self.img_files = sorted(list(p.glob('*')))
        else:
            for l in range(len(self.CLS_WEATHER)):
                p = Path(data_root) / 'splits' / self.CLS_WEATHER[l] / f'{split}.txt'
                if not p.is_file():
                    raise Exception(f'{p} does not exist or is not a file')
                with open(p, 'r') as f:
                    fnames = [Path(data_root) / 'images' / split / line.strip() for line in f]
                    self.img_files.extend(fnames)
                    self.labels.extend([l] * len(fnames))
                assert self.img_files, 'No images found'
                assert len(self.img_files) == len(self.labels), \
                    f'The number of labels {len(self.img_files)} does not match the number of images {len(self.labels)}.'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        if self.phase == 'test':
            img_path, target = self.img_files[idx], -1
        else:
            img_path, target = self.img_files[idx], self.labels[idx]

        img = cv2.imread(str(img_path))
        assert img is not None, 'Image Not Found ' + img_path

        h0, w0 = img.shape[:2]  # orig hw

        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        img, _, _ = letterbox(img, self.img_size, auto=False, scaleup=self.augment)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), target, img_path.name


def get_dataloader(opt, data_root, split, phase='train', shuffle=False):
    from DILAM.utils.torch_utils import torch_distributed_zero_first
    with torch_distributed_zero_first(-1):
        ds = ImageDataset(opt, data_root, split, phase)
    loader = DataLoader(ds, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.workers,
                        pin_memory=True, drop_last=False)

    return loader
