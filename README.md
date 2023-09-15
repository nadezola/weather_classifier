# Weather Classifier
Image Weather Classification with [DILAM](https://arxiv.org/abs/2305.18953)

## Installation on Ubuntu 20.04
1. Clone the repository recursively:
```bash
git clone --recurse-submodules https://github.com/nadezola/weather_classifier.git
```
* If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. We recommend to use Python3.8 virtual environment with `requirements.txt`

```bash
# apt install python3.8-venv
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3. Working directory is the root of the repository.

## Dataset preparation
