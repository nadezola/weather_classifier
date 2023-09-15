"""
Data pre-processing part:
    Splits the train/val datasets by weather conditions:
    Clear / Fog / Rain / Snow
Usage (from root dir):
    python data_preprocessing/data_splitting.py --data_root <path/to/data> --weather_lbls <path/to/weather_labels>
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', action='store', default='data', help='Path to data root')
    parser.add_argument('--weather_lbls', action='store', default='data/weather_labels.csv',
                        help='Path to weather labels')

    args = parser.parse_args()
    return args


def make_res_dirs(res_root):
    clear_dir = res_root / 'clear'
    fog_dir = res_root/'fog'
    rain_dir = res_root/'rain'
    snow_dir = res_root/'snow'

    if not clear_dir.exists():
        clear_dir.mkdir()
    if not fog_dir.exists():
        fog_dir.mkdir()
    if not rain_dir.exists():
        rain_dir.mkdir()
    if not snow_dir.exists():
        snow_dir.mkdir()


def split_set(data, weather_lbls):
    clear_hashes = weather_lbls.loc[weather_lbls['Weather'] == ('clear'), 'Hash'].values
    fog_hashes = weather_lbls.loc[weather_lbls['Weather'] == ('fog'), 'Hash'].values
    rain_hashes = weather_lbls.loc[weather_lbls['Weather'] == ('rain'), 'Hash'].values
    snow_hashes = weather_lbls.loc[weather_lbls['Weather'] == ('snow'), 'Hash'].values

    clear_files = []
    fog_files = []
    rain_files = []
    snow_files = []
    other_files = []

    for fname in data.values:
        fhash = Path(fname[0]).stem.split('_')[0]
        if fhash in clear_hashes:
            clear_files.extend(fname)
        elif fhash in fog_hashes:
            fog_files.extend(fname)
        elif fhash in rain_hashes:
            rain_files.extend(fname)
        elif fhash in snow_hashes:
            snow_files.extend(fname)
        else:
            other_files.extend(fname)

    return clear_files, fog_files, rain_files, snow_files, other_files


if __name__ == '__main__':
    args = parse_args()
    data_root = Path(args.data_root)
    train_split_file = data_root/'train.txt'
    val_split_file = data_root/'val.txt'
    weather_lbl_file = args.weather_lbls
    res_root = Path('data/splits')
    make_res_dirs(res_root)

    train_split = pd.read_csv(train_split_file, header=None)
    val_split = pd.read_csv(val_split_file, header=None)
    weather_lbls = pd.read_csv(weather_lbl_file, sep=';')

    for phase in ['train', 'val']:
        if phase == 'train':
            data = train_split
        if phase == 'val':
            data = val_split

        clear_files, fog_files, rain_files, snow_files, other_files = split_set(data, weather_lbls)
        with open(res_root/'clear'/f'{phase}.txt', 'w') as f:
            f.write('\n'.join(clear_files))
        with open(res_root/'fog'/f'{phase}.txt', 'w') as f:
            f.write('\n'.join(fog_files))
        with open(res_root/'rain'/f'{phase}.txt', 'w') as f:
            f.write('\n'.join(rain_files))
        with open(res_root / 'snow' / f'{phase}.txt', 'w') as f:
            f.write('\n'.join(snow_files))

        # Statistics
        total_images = len(clear_files) + len(fog_files) + len(rain_files) + len(snow_files) + len(other_files)
        with open(res_root/f'statistics_{phase}.txt', 'w') as f:
            f.write(f'clear: {len(clear_files)} ({(len(clear_files) / total_images)*100:.2f}%)\n')
            f.write(f'fog: {len(fog_files)} ({(len(fog_files) / total_images)*100:.2f}%)\n')
            f.write(f'rain: {len(rain_files)} ({(len(rain_files) / total_images)*100:.2f}%)\n')
            f.write(f'snow: {len(snow_files)} ({(len(snow_files) / total_images)*100:.2f}%)\n')
            f.write(f'others: {len(other_files)} ({(len(other_files) / total_images) * 100:.2f}%)\n')
            f.write('----------------------\n')
            f.write(f'total: {total_images}\n')




