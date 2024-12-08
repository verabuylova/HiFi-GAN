# Text-to-Speech (TTS) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an implemetation of HiFi-GAN with PyTorch.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=hifigan
```

## How To Download 
To download pretrained model, use:
```bash
wget
```
## How To Evaluate
To synthesize an audio from audio, your dataset should follow this structure:
```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.wav
    ├── UtteranceID2.wav
    .
    .
    .
    └── UtteranceIDn.wav
```

To get predictions, run
```bash
python3 synthesize.py -cn=from_audio HYDRA_CONFIG_ARGUMENTS
```

To synthesize an audio from text, your dataset should follow this structure:
```bash
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```
To get predictions, run
```bash
python3 synthesize.py -cn=from_text HYDRA_CONFIG_ARGUMENTS
```

Wandb report is available [here](https://wandb.ai/verabuylova-nes/hifigan/reports/HiFi-GAN--VmlldzoxMDUwOTI1NQ).

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)