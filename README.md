# solar-panel-damage-classification

## Descriprion

Pipeline for training a binary classifier designed to identify damaged solar panels, based on [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

## Installation

To get started with the project, you need to install and initialize the [poetry](https://python-poetry.org/docs/) environment.
```
pip install poetry
poetry install 
```
This will automatically create a new virtual environment and install the necessary dependencies.

## Usage

### Prepare data

Put your data in *./data/*. It should be a folder with images of solar panels and *.csv* tables with class labels for training, validation, and (optional) example datasets.

The data in the tables has to be in the following format:
```
image,label
images/001.png,0.0
...
```

Next, you need to create a *./conf/data/data.yaml* config, with the following content:
```
# @package data
image_folder: path_to_project/data
train_path: path_to_project/data/train.csv
valid_path: path_to_project/data/val.csv
test_path: path_to_project/data/test.csv
tb_viz_path: path_to_project/data
num_workers: 8
batch_size: 32
height: 300
width: 300

# optional
example_path: path_to_project/data/example.csv
```

### Start training

Before you start working, do not forget to activate the virtual environment, if you have not done that yet:
```
poetry shell
```

To start learning, use the CLI:
```
python train.py
```

You can change the training parameters directly from the command line, by to [hydra](https://hydra.cc/) configs:
```
python train.py data.batch_size=64
```
Or to run in multirun mode:
```
python train.py training.lr=0.001,0.01 training.weight_decay=0.1,0.01 -m
```

### Track metrics

All training runs are logged by [MLflow](https://mlflow.org/docs/latest/index.html) and [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). 

To run MLfow UI go to the working directory of the project, use the CLI `mlflow ui` and view it at *http://localhost:5000*.

To run TensorBoard UI go to the working directory of the project, use the CLI `tensorboard --logdir=/outputs` and view it at *http://localhost:6006*.

The local output of runs is stored in *./outputs/*.
