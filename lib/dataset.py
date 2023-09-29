from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from DILAM.utils.datasets import letterbox


class ImageDataset(Dataset):
    def __init__(self, opt, data_root, phase):
        self.img_size = opt.img_size
        self.CLS_WEATHER = opt.CLS_WEATHER
        self.augment = opt.augment
        self.phase = phase
        self.img_files = []
        self.labels = []

        if self.phase == 'test':
            p = Path(data_root) / 'images' / self.phase
            self.img_files = sorted(list(p.glob('*')))
        else:
            for l in range(len(self.CLS_WEATHER)):
                p = Path(data_root) / 'splits' / self.CLS_WEATHER[l] / f'{self.phase}.txt'
                if not p.is_file():
                    raise Exception(f'{p} does not exist or is not a file')
                with open(p, 'r') as f:
                    fnames = [Path(data_root) / 'images' / self.phase / line.strip() for line in f]
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
        return torch.from_numpy(img), target


def get_dataloader(opt, data_root, phase, shuffle=False):
    from DILAM.utils.torch_utils import torch_distributed_zero_first
    with torch_distributed_zero_first(-1):
        ds = ImageDataset(opt, data_root, phase)
    loader = DataLoader(ds, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.workers,
                        pin_memory=True, drop_last=False)

    return loader
