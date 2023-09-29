# Weather Classifier
Image Weather Classification with [DILAM](https://arxiv.org/abs/2305.18953)

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
data
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

| Hash              | Weather |
|-------------------|---------|
| vienna20181007_f0 | fog     |
| vienna20181007_f1 | fog     |

### Splitting by Weather Condition:
To train the Weather Classifier, you need to split train and val datsets by weather conditions:
```
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
```
* To split, run:
```bash
  python data_splitting.py --data_root <path/to/data> --weather_lbls <path/to/weather_labels> --res <path/to/folder/where/save/results>
```