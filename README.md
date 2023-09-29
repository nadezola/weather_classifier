# Weather Classifier
Image Weather Classification with inspiration from [DILAM](https://arxiv.org/abs/2305.18953).

## Installation on Ubuntu 20.04
* Clone the repository recursively:
```bash
git clone --recurse-submodules https://github.com/nadezola/weather_classifier.git
```
* If you already cloned and forgot to use `--recurse-submodules` you can run: 
```bash
git submodule update --init
```

* We recommend to use Python3.8 virtual environment with `requirements.txt`

```bash
# apt install python3.8-venv
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
* Working directory is the root of the repository.

## Dataset preparation
### Data structure:
```
data_root
|
├── images
|      ├─── test
|      |     ├─── *.jpg
|      |
|      ├─── train
|      |     ├─── i*.jpg
|      |
|      └─── val
|            ├─── *.jpg
|
├── labels
|      └─── test
|      |     ├─── *.txt
|      |
|      └─── train
|      |     ├─── *.txt
|      |
|      └─── val
|            ├─── *.txt
|
├── splits (initially empty folder)
|
├── test.txt
├── train.txt
├── val.txt
|
└── weather_labels.csv
```
* `test.txt`, `train.txt`, `val.txt` contain a list of images defining the train/val/test splits.
* `weather_labels.csv` contains annotations of weather conditions. Example:

| Image Name | Weather |
|----------------------|---------|
| vienna20181007_f0    | fog     |
| vienna20181007_f1    | fog     |

### Splitting by Weather Condition:
To train the Weather Classifier for 4 classes _**clear, fog, rain, snow**_, you need to split train and val datsets by weather conditions:
```
data_root
|
...
|
├── splits (initially empty folder)
|      └─── clear
|      |     ├─── train.txt
|      |     └─── val.txt
|      |
|      └─── fog
|      |     ├─── train.txt
|      |     └─── val.txt
|      |
|      └─── rain
|      |     ├─── train.txt
|      |     └─── val.txt
|      |
|      └─── snow
|            ├─── train.txt
|            └─── val.txt
...
|
```
* We provide example code to split the data by weather conditions:
```bash
  python data_splitting.py --data_root <path/to/data_root> 
                           --weather_lbls <path/to/weather_labels>
                           --res_dir <path/to/folder/where/save/results>
```

## Train
* Configure `opt.py` file
* Pre-train the YOLOv3 on clear weather dataset for your object detection task and put the pre-trained model in `checkpoints/` folder
* **or** use our YOLOv3 model pre-trained on [KITTI object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) for 8 classes: `checkpoints/YOLOv3_clear_kitti_pretrained.pt`
* Run:
```bash
python main.py --data_root <path/to/data/root/> 
               --phase 'train'
               --res_dir <path/to/folder/where/save/results>
```
