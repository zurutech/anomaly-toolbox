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

Anomaly Toolbox Powered by GANs

## Installation

```console
pip install anomaly-toolbox
```

## Usage

```
anomaly_toolbox --experiment {choose one of the available experiments}
```

## Features

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

## Testing

Run complete test suite + linting:

```console
tox
```

## TODOS

- [x] Implement Models
- [x] Implement Losses
- [x] Implement Datasets (basics)
- [ ] Implement Training
    - [x] Configure Experiments and HPs on TB
    - [x] GANomaly
    - [x] BiGAN
    - [x] AnoGAN
- [ ] Export SavedModel
- [ ] Add a dedicated predictor module
  - [ ] Add AUC metric (or something like that)
  - [ ] Benchmark!
- [x] Add a proper CLI
  - [ ] Expand the CLI by allowing the passing of a file with an Experiment to run
- [ ] Implement Datasets (all)
- [ ] Implement More Datasets
  - [ ] MNIST corrupted
  - [ ] Chest X-RAY
  - [ ] NVIDIA Industrial Semantic Segmentation
- [ ] Implement more models
  - [ ] FastAnoGAN
  - [ ] DeScarGAN
  - [ ] Extend with new VAE?
  - [ ] Extend with new Diffusion Model?
- [ ] Tests
- [ ] Docs

## Credits

This package was created with [Cookiecutter] and the [zurutech/cookiecutter-pypackage] project template.

Requirements are structured according to [zurutech/styleguide] and should be handled with [pip-tools] or [reqompyler].

[Cookiecutter]: https://github.com/audreyr/cookiecutter
[pip-tools]: https://github.com/jazzband/pip-tools
[reqompyler]: https://github.com/zurutech/reqompyler
[zurutech/cookiecutter-pypackage]: https://github/zurutech/cookiecutter-pypackage
[zurutech/styleguide]: https://github.com/zurutech/styleguide/python.md
