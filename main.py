import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data', help='path/to/data_root')
    parser.add_argument('--ckpt_path', default='checkpoints/yolov3_clear_pretrained.pt', help='path/to/model_checkpoint.pt')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pass

