# RLCC Re-ID Baseline

This is the RLCC Re-id baseline. The codebase is modified from OpenUnReID project.

## Installation

### Requirements

+ ubuntu 16.04+
+ Python 3.5+
+ PyTorch 1.1 or higher
+ CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

+ OS: Ubuntu 16.04
+ Python: 3.6/3.7
+ PyTorch: 1.1/1.5/1.6
+ CUDA: 9.0/11.0

### Install OpenUnReID

**a.** Create a conda virtual environment and activate it.
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

**b.** Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```shell
conda install pytorch torchvision -c pytorch
```

**c.** Install the dependent libraries.
```shell
cd RLCC_Baseline
pip install -r requirements.txt
```

**d.** Install `openunreid` library.
```shell
python setup.py develop
```

**e.** Support [AutoAugment](https://arxiv.org/abs/1805.09501). (optional)

You may meet the following error when using `DATA.TRAIN.is_autoaug=True` in config files,
>AttributeError: Can't pickle local object 'SubPolicy.__init__.<locals>.<lambda>'

To solve it, you need to replace `multiprocessing` with `multiprocess` in `torch.multiprocessing` (generally found in `$CONDA/envs/open-mmlab/lib/python3.7/site-packages/torch/multiprocessing/`), e.g.
```shell
# refer to https://github.com/DeepVoltaire/AutoAugment/issues/16
import multiprocess as multiprocessing
from multiprocess import *
```


### Prepare datasets

It is recommended to symlink your dataset root to `OpenUnReID/datasets`. If your folder structure is different, you may need to change the corresponding paths (namely `DATA_ROOT`) in config files.

Download the datasets from
+ DukeMTMC-reID: [[Google Drive]](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view)
+ Market-1501-v15.09.15: [[Google Drive]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
+ MSMT17_V1: [[Home Page]](https://www.pkuvmc.com/dataset.html) (request link by email the holder)
+ VehicleID_V1.0: [[Home Page]](https://www.pkuml.org/resources/pku-vehicleid.html) (request link by email the holder)
+ AIC20_ReID_Simulation (VehicleX): [[Home Page]](https://www.aicitychallenge.org/2020-track2-download/) (request password by email the holder)
+ VeRi_with_plate: [[Home Page]](https://github.com/JDAI-CV/VeRidataset#2-download) (request link by email the holder)

Save them under
```shell
OpenUnReID
└── datasets
    ├── dukemtmcreid
    │   └── DukeMTMC-reID
    ├── market1501
    │   └── Market-1501-v15.09.15
    ├── msmt17
    │   └── MSMT17_V1
    ├── personx
    │   └── subset1
    ├── vehicleid
    │   └── VehicleID_V1.0
    ├── vehiclex
    │   └── AIC20_ReID_Simulation
    └── veri
        └── VeRi_with_plate
```



## Getting Started

The training and testing scripts can be found in `OpenUnReID/tools`. We use 4 GPUs for training and testing, which is considered as a default setting in the scripts. You can adjust it (e.g. `${GPUS}`, `${GPUS_PER_NODE}`) based on your own needs.

### Test

#### Testing commands

+ Distributed testing with multiple GPUs:
```shell
bash tools/dist_test.sh ${RESUME}
```

+ Testing with a single GPU:
```shell
GPUS=1 bash tools/dist_test.sh ${RESUME}
```

#### Arguments

+ `${RESUME}`: model for testing, e.g. `../logs/20210101/market1501/model_best.pth`.

#### Configs

+ Test with different datasets, e.g.
```shell
TEST:
  datasets: ['market1501',] # arrange the names in a list
```
+ Add re-ranking post-processing, e.g.
```shell
TEST:
  rerank: True # default: False
```
+ Save GPU memory but with a lower speed,
```shell
TEST:
  dist_cuda: False # use CPU for computing distances, default: True
  search_type: 3 # use CPU for re-ranking, default: 0 (1/2 is also for GPU)
```


### Train

#### Training commands

+ Training with single node multiple GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh ${WORK_DIR} 
```

#### Arguments

+ `${WORK_DIR}`: folder for saving logs and checkpoints, e.g. `20210101/market1501`, the absolute path will be `LOGS_ROOT/${WORK_DIR}` (`LOGS_ROOT` is defined in config files).


#### Configs

+ Flexible architectures,
```shell
MODEL:
  backbone: 'resnet50' # or 'resnet101', 'resnet50_ibn_a', etc
  pooling: 'gem' # or 'avg', 'max', etc
  dsbn: True # domain-specific BNs, critical for domain adaptation performance
```

+ Ensure reproducibility (may cause a lower speed),
```shell
TRAIN:
  deterministic: True
```

+ Dataset Config,
the conventional USL task, e.g. unsupervised market1501
```shell
TRAIN:
  # arrange the names in a dict, {DATASET_NAME: DATASET_SPLIT}
  datasets: {'market1501': 'trainval'}
  # val_set of 'market1501' will be used for validation
  val_dataset: 'market1501'
```

+ Mixed precision training
```shell
TRAIN:
  amp: True # mixed precision training for PyTorch>=1.6
```
