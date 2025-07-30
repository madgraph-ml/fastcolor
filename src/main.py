import os
import time
import shutil
from distutils.dir_util import copy_tree
import glob
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.plots_utils import Metric
from .utils.logger import setup_logging
from .utils.mlflow import mlflow, log_mlflow, LOGGING_ENABLED
from .datasets.gluons import gg_ng, gg_ddbarng
from .datasets.dataset import compute_observables
from collections import defaultdict
from .models.models import Model, MLP, Transformer
from .models.lgatr import LGATr

# from lgatr import LGATr as LGATr_legacy
from .plots import Plots
import torch


def init_logger(run_dir):
    """
    Initialize the logger. If we are working on a previous run, output to "out_{previos_run_number+1}.log"
    """
    log_files = glob.glob(os.path.join(run_dir, "out_*.log"))
    log_files.sort()
    if len(log_files) > 0:
        run_number = int(log_files[-1].split("_")[-1].split(".")[0])
        run_number += 1
    else:
        run_number = 0
    log_file_path = os.path.join(run_dir, f"out_{run_number:02d}.log")
    return setup_logging(log_file_path)


def main(cfg: DictConfig):
    base_dir = hydra.utils.get_original_cwd()

    if cfg.run.type == "train":
        if not cfg.train.warm_start:
            # Initialize run directory
            results_dir = os.path.join(base_dir, "results")
            run_name = (
                cfg.model.type
                + "/"
                + datetime.now().strftime("%m%d_%H%M%S")
                + "-"
                + cfg.run.name
                if cfg.run.name is not None
                else cfg.model.type + "/" + datetime.now().strftime("%m%d_%H%M%S")
            )
            run_dir = os.path.join(results_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
        else:
            # Load already existing run directory
            run_dir = cfg.run.path
            old_dir = os.path.join(run_dir, "old")
            os.makedirs(old_dir, exist_ok=True)
            for item in os.listdir(run_dir):
                item_path = os.path.join(run_dir, item)
                if (
                    (
                        os.path.isfile(item_path)
                        and item.startswith("out_")
                        and item.endswith(".log")
                    )
                    or item == "model"
                    or (
                        os.path.isfile(item_path)
                        and item.startswith("config")
                        and item.endswith(".yaml")
                    )
                ):
                    continue
                if "old" in item:
                    continue
                os.rename(item_path, os.path.join(old_dir, item))

        # This is to keep plot as predefined run type
        cfg_to_save = OmegaConf.to_container(cfg, resolve=True)
        cfg_to_save["run"]["path"] = run_dir
        cfg_to_save["run"]["type"] = "plot"
        config_file = os.path.join(run_dir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.create(cfg_to_save)))

        shutil.copytree(
            "src",
            os.path.join(run_dir, "src"),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "*__pycache__",
                "*egg-info",
                "playground",
                "template_files",
                "*__init__.py",
            ),
        )
    elif cfg.run.type == "plot":
        # Load already existing run directory
        run_dir = cfg.run.path
        old_dir = os.path.join(run_dir, "old")
        os.makedirs(old_dir, exist_ok=True)
        for item in os.listdir(run_dir):
            item_path = os.path.join(run_dir, item)
            if (
                (
                    os.path.isfile(item_path)
                    and item.startswith("out_")
                    and item.endswith(".log")
                )
                or item == "model"
                or (
                    os.path.isfile(item_path)
                    and item.startswith("config")
                    and item.endswith(".yaml")
                )
            ):
                continue
            if "old" in item:
                continue
            if not os.path.exists(os.path.join(old_dir, item)):
                os.rename(item_path, os.path.join(old_dir, item))
            else:
                if os.path.isdir(item_path):
                    shutil.rmtree(os.path.join(old_dir, item))
                else:
                    os.remove(os.path.join(old_dir, item))
                os.rename(item_path, os.path.join(old_dir, item))

    else:
        raise NotImplementedError(f"Run type {cfg.run.type} not recognised")

    logger = init_logger(run_dir)
    try:
        run(logger, run_dir, cfg)
    except Exception as e:
        logger.error(e, exc_info=True)


def run(logger, run_dir, cfg: DictConfig):
    """
    Run training and/or plotting of the model with the given parameters.
    """

    ### INITALIZE RUN ###

    logger.info(f"Starting {cfg.run.type} run in {run_dir}")
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logger.info(f"Device {device}")
    torch.set_default_dtype(getattr(torch, cfg.backend.get("torch_dtype", "float64")))

    ### INITIALIZE MLFLOW ###
    if LOGGING_ENABLED and device == "cuda" and cfg.run.type == "train":
        logger.info(f"Setting up MLflow tracking")
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(f"{cfg.dataset.process}")
        mlflow.start_run(
            run_name=cfg.model.type + cfg.run.name
            if cfg.run.name is not None
            else run_dir
        )
        # flatten and log top‐level params
        for key, val in {
            **cfg.train,
            **cfg.model,
            **cfg.dataset,
            **cfg.dataset.parameterization.naive,
            **cfg.dataset.parameterization.lorentz_products,
            **cfg.dataset.preprocessing,
            **{
                "run_type": cfg.run.type,
                "run_name": cfg.run.name,
                "dataset": cfg.dataset,
            },
        }.items():
            log_mlflow(logger, key, str(val), kind="param")

    # INITIALIZE DATASET AND PREPROCESSING ###
    logger.info(f"Dataset: {cfg.dataset.process}")
    # if cfg.model.type == "LGATr":
    #     cfg.dataset.parameterization.naive.use = True
    #     cfg.dataset.parameterization.lorentz_products.use = False
    param_names = [
        p for p in cfg.dataset.parameterization if cfg.dataset.parameterization[p].use
    ]
    logger.info(f"    Using parameterization(s): {', '.join(param_names)}")

    dataset = eval(cfg.dataset.type)(logger, cfg.dataset)
    dataset.apply_preprocessing()
    dataset.init_observables()

    ### INITIALIZE MODEL AND DATALOADERS ###
    dims_out = 2 if cfg.train.get("loss", "MSE") == "heteroschedastic" else 1
    dims_in = len(dataset.input_channels) - 1
    logger.info(
        f"Building model {cfg.model.type} with dims_in = {dims_in}, and dims_out = {dims_out}. Loss = {cfg.train.get('loss', 'heteroschedastic')}"
    )
    model_path = os.path.join(run_dir, "model")

    model = eval(cfg.model.type)(
        logger=logger,
        process=cfg.dataset.process,
        cfg=cfg,
        dims_in=dims_in,
        helicity_dict_size=dataset.helicity_dict_size
        if hasattr(dataset, "helicity_dict_size")
        else None,
        dims_out=dims_out,
        model_path=model_path,
        device=device,
    ).to(
        device
    )  # if cfg.model.type != "LGATr_legacy" else LGATr_legacy(
    #     in_mv_channels = cfg.model["in_mv_channels"],
    #     out_mv_channels = cfg.model["out_mv_channels"],
    #     hidden_mv_channels = cfg.model["hidden_mv_channels"],
    #     in_s_channels = cfg.model.get("in_s_channels", None),
    #     out_s_channels = cfg.model.get("out_s_channels", None),
    #     hidden_s_channels = cfg.model.get("hidden_s_channels", None),
    #     attention = cfg.model["attention"],
    #     mlp = cfg.model["mlp"],
    #     num_blocks = cfg.model.get("num_blocks", 10),
    #     reinsert_mv_channels = cfg.model.get("reinsert_mv_channels", None),
    #     reinsert_s_channels = cfg.model.get("reinsert_s_channels", None),
    #     checkpoint_blocks = False,
    #     dropout_prob = cfg.model.get("dropout_prob", None),
    #     double_layernorm = cfg.model.get("double_layernorm", False),
    # ).to(device)
    model.name = cfg.model.type
    logger.info(
        f"    Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    logger.info(f"    Dropout rate: {cfg.model.get('dropout', 0.1)}")
    if LOGGING_ENABLED:
        log_mlflow(
            logger,
            "num_parameters",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            kind="param",
        )
    model.init_dataloaders(dataset)

    if cfg.run.type == "train":
        # TRAIN MODEL
        if not cfg.train.warm_start:
            os.makedirs(model_path, exist_ok=True)
        else:
            pass
        model.train()
        if cfg.evaluate.get("evaluate_best", False):
            logger.info(f"Loading best model from {model_path}/best.pth")
            try:
                model.load("best")
            except FileNotFoundError:
                logger.warning(
                    f"Best model not found in {model_path}/best.pth. Loading final model instead."
                )
                model.load("final")
    elif cfg.run.type == "plot":
        if cfg.evaluate.get("model_name", None) is not None:
            model_name = cfg.evaluate.model_name
            try:
                model.load(model_name)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Model {model_name} not found in {model_path}/{model_name}.pth. Please train the model first."
                )
        else:
            model_name = (
                "final" if not cfg.evaluate.get("evaluate_best", False) else "best"
            )
            if model_name == "best":
                try:
                    model.load("best")
                except FileNotFoundError:
                    logger.warning(
                        f"Best model not found in {model_path}/best.pth. Loading final model instead."
                    )
                    model.load("final")
            else:
                model.load("final")

    ### EVALUATE MODEL ###
    dataset.predicted_factors_ppd = {}
    dataset.predicted_factors_raw = {}
    model.dataset_loss = defaultdict(dict)
    model.evaluation_time = defaultdict(dict)
    for k in ["trn", "tst", "val"]:
        t0 = time.time()
        dataset.predicted_factors_ppd[k] = model.evaluate(split=k)
        dataset.apply_preprocessing(reverse=True, split=k)
        model.evaluation_time[k] = time.time() - t0
        logger.info(
            f"    Evaluation time for {k} set: {model.evaluation_time[k]:.5f} seconds"
        )
        if cfg.evaluate.save_samples:
            os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
            torch.save(
                dataset.predicted_factors_ppd[k],
                os.path.join(run_dir, f"samples/predicted_factors_{k}.pt"),
            )
        if cfg.evaluate.save_data:
            os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
            torch.save(
                dataset.events[k],
                os.path.join(run_dir, f"samples/events_{k}.pt"),
            )

    # Compute dataset loss for each split
    for k in ["trn", "tst", "val"]:
        model.compute_dataset_loss(
            dataset.predicted_factors_raw[k],
            dataset.events[k][:, -1],
            split=k,
        )
    # ### COMPUTE OBSERVABLES ###
    logger.info("Computing observables")
    compute_observables(dataset, split="tst", include_ppd=True)
    metrics_dict = defaultdict(lambda: defaultdict(dict))
    for k in ["trn", "tst", "val"]:
        for m in model.dataset_loss.keys():
            metrics_dict[k][m]["loss"] = Metric(
                name=f"{k} ({m}) loss",
                value=model.dataset_loss[m][k],
                tex_label=rf"\text{{loss}}",
                unit="",
            )
            metrics_dict[k][m]["eval_time"] = Metric(
                name=f"{k} eval_time",
                value=model.evaluation_time[k],
                tex_label=rf"t_{{\text{{eval}}}}",  # rf"\text{{{k}}}\ (\text{{{m}}})\ t_{{\text{{eval}}}}",
                format="{:.3f}",
                unit="s",
            )

    ### MAKE PLOTS ###
    logger.info(f"Starting plots")
    plots = Plots(
        logger,
        dataset,
        model.losses if hasattr(model, "losses") else None,
        metrics=metrics_dict,
        process=cfg.dataset.process,
        regress=cfg.dataset.get("regress", "r"),
        debug=False,
        model_name=model.name,
        loss_name=cfg.train.get("loss", "MSE"),
    )

    percentage_of_ratio_data = (
        99.0  # showing 99% to avoid the massive (very few) outliers
    )

    if cfg.evaluate.get("save_lines", False):
        os.makedirs(os.path.join(run_dir, "pkl"), exist_ok=True)
    if hasattr(model, "losses"):
        logger.info(f"    Plotting train metrics")
        plots.plot_train_metrics(os.path.join(run_dir, f"train_metrics.pdf"), logy=True)
    # logger.info(f"    Plotting observables")
    # plots.plot_observables(os.path.join(run_dir, f"observables.pdf"))
    # logger.info(f"    Plotting ppd observables")
    # plots.plot_observables_ppd(os.path.join(run_dir, f"observables_ppd.pdf"))
    logger.info(f"    Plotting regressed factors and ratio correlations")
    for k in ["tst", "val", "trn"]:
        for ppd_flag, ppd_s in zip([False, True], ["", "_ppd"]):
            logger.info(
                f"        Plotting {k} set and { {True : 'ppd', False : 'raw'}[ppd_flag] }..."
            )
            plots.plot_weights(
                os.path.join(run_dir, f"factors{ppd_s}_{k}.pdf"),
                split=k,
                ppd=ppd_flag,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=os.path.join(run_dir, "pkl", f"factors{ppd_s}_{k}.pkl")
                if cfg.evaluate.get("save_lines", False)
                else None,
                fix_bins=cfg.evaluate.get("save_lines", False),
            )
            plots.plot_2d_correlations(
                os.path.join(run_dir, f"ratio_corr{ppd_s}_{k}.pdf"),
                split=k,
                ppd=ppd_flag,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=os.path.join(run_dir, "pkl", f"ratio_corr{ppd_s}_{k}.pkl")
                if cfg.evaluate.get("save_lines", False)
                else None,
                fix_bins=cfg.evaluate.get("save_lines", False),
            )
        plots.plot_amplitudes_closure_test(
            cfg,
            os.path.join(run_dir, f"FCvsrLC_{k}.pdf"),
            split=k,
            ppd=False,
            pickle_file=os.path.join(run_dir, "pkl", f"FCvsrLC_{k}.pkl")
            if cfg.evaluate.get("save_lines", False)
            else None,
            metrics=None,
        )

    if device == "cuda":
        max_used = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        currently_used = total - free
        logger.info(
            f"GPU RAM info: currently_used = {currently_used/1e9:.2f} GB, peak_used = {max_used/1e9:.2f} GB, total available= {total/1e9:.2f} GB"
        )
    logger.info("Run finished")
