<h1 align="center">MadRecolor</h1>

<h2 align="center">Machine-Learned Leading-Color Amplitude Reweighting for MadGraph</h2>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

**MadRecolor** is a Python-based tool designed to perform **leading-color amplitude reweighting** using machine learning techniques within the **MadGraph** framework. It enables efficient reweighting of matrix elements by leveraging neural networks to approximate color weight factors, improving computational efficiency in high-energy physics simulations.

## Installation
To install **MadRecolor**, clone the repository and install dependencies:

```bash
git clone git@github.com:madgraph-ml/madrecolor.git
# then install in dev mode
cd madrecolor
pip install --editable .
```

## Dependencies
- Python 3.x
- NumPy 1.x
- Black
- PyTorch 2.x
- Hydra

## Usage

Training a model:
```sh
python run.py -cn config/config_file.yaml
```
A folder will be created in `madrecolor/results` with the name of the model employed for better traceability. Inside, a new subfolder will appear with the date and time of the run, and will contain log files, the config of the run, the trained model and plots.

To regenerate plots for a trained model, it is sufficient to specify the config path `-cp` stored in the run path, and the name of the config file `cn`. For example:
```sh
python run.py -cn config_from_run1 -cp results/my_model/YYYYMMDD_HHMMSS-run1
```
To warm-start a pre-trained model and continue the training. Just specify the path and override the `run.type` and `train.warm_start` settings:
```sh
python run.py -cn config_from_run1 -cp results/my_model/YYYYMMDD_HHMMSS-run1 run.type=train train.warm_start=true
```