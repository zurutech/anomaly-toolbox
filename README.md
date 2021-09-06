# Anomaly Toolbox![Python - Version](https://img.shields.io/pypi/pyversions/anomaly_toolbox.svg)
![PyPy - Version](https://badge.fury.io/py/anomaly_toolbox.svg)
![PyPI - License](https://img.shields.io/pypi/l/anomaly_toolbox.svg)
![Anomaly Toolbox - Badge](https://img.shields.io/badge/package-anomaly-toolbox-brightgreen.svg)
[![Build Status](https://img.shields.io/travis/zurutech/anomaly-toolbox.svg)](https://travis-ci.org/zurutech/anomaly-toolbox)
[![Documentation Status](https://readthedocs.org/projects/anomaly-toolbox/badge/?version=latest)](https://anomaly-toolbox.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/zurutech/anomaly-toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/zurutech/anomaly-toolbox)
[![CodeFactor](https://www.codefactor.io/repository/github/zurutech/anomaly-toolbox/badge)](https://www.codefactor.io/repository/github/zurutech/anomaly-toolbox)![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
[![Code Style - Zuru](https://img.shields.io/badge/codestyle-zuru-red)](https://github.com/zurutech/styleguide)
[![Black - Badge](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

## Description

Anomaly Toolbox Powered by GANs. This is the accompanying toolbox for the paper "**A 
Survey on GANs for Anomaly Detection**" (https://arxiv.org/pdf/1906.11632.pdf).

The toolbox is meant to be used by the user to explore the performance of different GAN based 
architectures (aka "**experiments**"). It also already provides some datasets to do experiments 
on: 
_MNIST_, 
_Corrupted 
MNIST_ and _Surface Cracks_ 
(https://www.kaggle.com/arunrk7/surface-crack-detection). 

Apart from the _Corrupted Mnist_ and _Surface Cracks_, the _MNIST_ dataset was provided because the 
original works extensively use it. All the architecture were tested on commonly used datasets such 
as _MNIST_, _FashionMNIST_, _CIFAR-10_, and _KDD99_. Some of them were even tested on more 
designated datasets, such as an X-Ray dataset that we cannot provide because of the impossibility 
of getting the data. 

The user can create their own dataset and use it to test the models.

## Quick Start

* First thing first, install the toolbox

```bash 
pip install anomaly-toolbox
```

Then you can choose what to run, for example:

* Run all the experiments using the pre-defined hyperparameters file _hparams.json_ and the _MNIST_ 
dataset:

```bash
anomaly-box.py --run-all True --hps-path path/to/config/hparams.json --dataset MNIST 
```

* Run a specified experiment, for example the GANomaly architecture, with hyperparameters tuning 
enabled, the pre-defined hyperparameters file _hparams.json_ and the _MNIST_ 
dataset:

```bash
anomaly-box.py --experiment GANomalyExperiment --hps-path path/to/config/hparams.json --dataset 
MNIST 
```

For any other information, feel free to check the help:

```bash 
anomaly-box.py --help
```

## Contribution

This work is completely open source, and we would appreciate any contribution to the code. Any merge 
request to enhance, correct or expand the work is welcome.

## Notes

The structures of the models inside the toolbox come from their respective papers. We have tried to 
respect them as much as possible. However, sometimes, due to implementation issues, we had to make 
some minor-ish changes. For this reason, you could find out that, in some cases, some features 
such as the number of layers, the size of kernels, or other such things may differ from the 
originals. 

However, you don't have to worry. The heart and purpose of the architectures have remained intact.

## Installation

```console
pip install anomaly-toolbox
```

## Usage

```
Options:
  --experiment [AnoGANExperiment|DeScarGANExperiment|EGBADExperiment|GANomalyExperiment]
                                  Experiment to run.
  --hps-path PATH                 When running an experiment, the path of the
                                  JSON file where all the hyperparameters are
                                  located.  [required]
  --tuning BOOLEAN                If you want to use hyperparameters tuning,
                                  use 'True' here. Default is False.
  --dataset TEXT                  The dataset to use. Can be a ready to use
                                  dataset, or a .py file that implements the
                                  AnomalyDetectionDataset interface
                                  [required]
  --run-all BOOLEAN               Run all the available experiments
  --help                          Show this message and exit.
```

## Datasets and Custom Datasets

The provided datasets are:

* MNIST 
* Corrupted Mnist
* Surface Crack (https://www.kaggle.com/arunrk7/surface-crack-detection)

and are automatically downloaded when the user makes a specific choice: ["MNIST", 
"CorruptedMNIST", "SurfaceCracks"].

The user can also add its own specific dataset. To do this, the new dataset should inherit from the `AnomalyDetectionDataset` abstract class implementing its own `configure` method. For a more detailed guide, the user can refer to the `README.md` file inside the `src/anomaly_toolbox/datasets` folder. Moreover, in the `examples` folder, the user can find a `dummy.py` module with the basic skeleton code to implement a dataset.

## References

- **GANomaly**:
    - Paper: https://arxiv.org/abs/1805.06725
    - Code: https://github.com/samet-akcay/ganomaly
- **EGBAD (BiGAN)**:
    - Paper: https://arxiv.org/abs/1802.06222
    - Code: https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection
- **AnoGAN**:
    - Paper: https://arxiv.org/abs/1703.05921
    - Code (not official): https://github.com/LeeDoYup/AnoGAN
    - Code (not official): https://github.com/tkwoo/anogan-keras
- **DeScarGAN**:
    - Paper: https://arxiv.org/abs/2007.14118
    - Code: https://github.com/JuliaWolleb/DeScarGAN
