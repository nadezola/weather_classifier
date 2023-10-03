"""
Data pre-processing part. Splits the train/val datasets by weather conditions:
    Clear (train.txt / val.txt)
    Fog (train.txt / val.txt)
    Rain (train.txt / val.txt)
    Snow (train.txt / val.txt)

Usage:
    python data_splitting.py --data_root <path/to/data/root> --weather_lbls <path/to/weather/labels> --res_dir <path/to/folder/where/to/save/results>
"""

import argparse
import pandas as pd
from pathlib import Path


def make_dir(dir):
    if not dir.exists():
        dir.mkdir(parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', action='store', default='data', help='Path to data root')
    parser.add_argument('--weather_lbls', action='store', default='data/weather_labels.csv',
                        help='Path to weather labels')
    parser.add_argument('--res_dir', action='store', default='data/splits',
                        help='Path to save split results')

    args = parser.parse_args()
    return args


def make_split_dirs(res_root):
    make_dir(res_root /'clear')
    make_dir(res_root/'fog')
    make_dir(res_root/'rain')
    make_dir(res_root/'snow')


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

    for f in data.values:
        fhash = Path(f[0]).stem.split('_')[0]
        fname = Path(f[0]).name
        if fhash in clear_hashes:
            clear_files.append(fname)
        elif fhash in fog_hashes:
            fog_files.append(fname)
        elif fhash in rain_hashes:
            rain_files.append(fname)
        elif fhash in snow_hashes:
            snow_files.append(fname)
        else:
            other_files.append(fname)

    return clear_files, fog_files, rain_files, snow_files, other_files


if __name__ == '__main__':
    args = parse_args()
    data_root = Path(args.data_root)
    res_root = Path(args.res_dir)
    weather_lbl_file = args.weather_lbls

    train_txt = data_root/'train.txt'
    val_txt = data_root/'val.txt'

    train_split = pd.read_csv(train_txt, header=None)
    val_split = pd.read_csv(val_txt, header=None)

    weather_lbls = pd.read_csv(weather_lbl_file, sep=';')

    make_split_dirs(res_root)

    for data in [train_split, val_split]:
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




