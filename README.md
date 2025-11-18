<h1 align="center">
  FAST<span style="color:#C20000">C</span><span style="color:#D69526">o</span><span style="color:#06793F">l</span><span style="color:#790679">o</span><span style="color:#07078A">r</span>
</h1>

<h2 align="center">Full-color Amplitude Surrogate Toolkit for QCD</h2>

<p align="center">
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
<a href="https://hydra.cc/"><img alt="Config style: Hydra" src="https://img.shields.io/badge/Hydra-1.2-78a9c2"></a>
<a href="https://github.com/psf/black"><img alt="Code style: Black" src="https://img.shields.io/badge/Black-22.3-000000.svg"></a>
<a href="https://mlflow.org"><img alt="MLOps: MLflow" src="https://img.shields.io/badge/MLflow-2.22.0-1388DB.svg"></a>
</p>

**FASTColor** is a Python-based tool designed to perform **leading-color amplitude reweighting** using machine learning techniques. It enables unweighted event generation at full-color accuracy by leveraging neural networks trained to reweight them from leading-color accuracy. This improves computational efficiency and speeds-up the generation of QCD processes.

## Installation
To install **FASTColor**, clone the repository and install dependencies:

```bash
git clone git@github.com:madgraph-ml/fastcolor.git
# then install in dev mode
cd madrecolor
pip install --editable .
```

## Dependencies
- Python 3.x
- NumPy 1.x
- Black
- PyTorch 2.x
- Hydra 1.3
- MLflow 2.22.0

## Usage

Training a model:
```sh
python run.py -cn config/config_file.yaml
```
A folder will be created in `results/` with the name of the model employed for better traceability. Inside, a new subfolder will appear with the date and time of the run, and will contain log files, the config of the run, the trained model and plots.

To regenerate plots for a trained model, it is sufficient to specify the config path `-cp` stored in the run path, and the name of the config file `cn`. For example:
```sh
python run.py -cn config_from_run1 -cp results/my_model/MMDD_HHMMSS-run1
```
To warm-start a pre-trained model and continue the training. Just specify the path and override the `run.type` and `train.warm_start` settings:
```sh
python run.py -cn config_from_run1 -cp results/my_model/MMDD_HHMMSS-run1 run.type=train train.warm_start=true
```
Models that one can use are organized in experiments trees, and each contain multiple runs. We use `mlflow` to track metrics during training and save them to a database object within the mlflow folder. A local mlflow web interface using port 4242 can be started with the command
```
mlflow ui --port 4242 --backend-store-uri sqlite:///mlruns/mlflow.db
```
Bear in mind that the memory one can use for UI/API calls is limited, and will not support the logging of large batches of metrics per call, so it is recommended to do the logging at the frequency of epochs and not higher.

## Citation

If you use this code or parts of it, please cite:

    @article{Villadamigo:2025our,
    author = "Villadamigo, Javier Mari{\~n}o and Frederix, Rikkert and Plehn, Tilman and Vitos, Timea and Winterhalder, Ramon",
    title = "{FASTColor -- Full-color Amplitude Surrogate Toolkit for QCD}",
    eprint = "2509.07068",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "TIF-UNIMI-2025-18",
    month = "9",
    year = "2025"}
