import os
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from utils.logger import setup_logging
from datasets.gluons import gg_ng, gg_qqbarng
from datasets.dataset import compute_observables
from models.models import Model, MLP, MDN
from models.lgatr import LGATr, LGATr_net, AmplitudeWrapper
from plots import Plots
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
        ### INITIALIZE DIRECTORIES AND LOAD PARAMS ###
        results_dir = os.path.join(base_dir, "results")
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + cfg.run.name
        run_dir = os.path.join(results_dir, run_name)

        os.makedirs(run_dir, exist_ok=True)
        # Save config for reproducibility
        with open(os.path.join(run_dir, "params.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    
    elif cfg.run.type == "plot":
        ### LOAD PARAMS FROM EXISTING RUN ###
        run_dir = cfg.run.path
    else:
        raise NotImplementedError(f"Run type {cfg.run.type} not recognised")
    

    ### INITIALIZE LOGGER ###
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
    logger.info(f"Device {device}")

    # INITIALIZE DATASET AND PREPROCESSING ###
    logger.info(f"Dataset: {cfg.dataset.process}")
    if cfg.model.type == "LGATr":
        cfg.dataset.parameterisation.naive.use = True
        cfg.dataset.parameterisation.lorentz_products.use = False
    param_names = [p for p in cfg.dataset.parameterisation if cfg.dataset.parameterisation[p].use]
    logger.info(f"Using parameterisation(s): {', '.join(param_names)}")

    dataset = eval(cfg.dataset.type)(logger, cfg.dataset)
    dataset.apply_preprocessing()
    dataset.init_observables()

    ### INITIALIZE MODEL AND DATALOADERS ###
    dims_out = 1
    dims_in = len(dataset.channels) - 1
    logger.info(
        f"Building model {cfg.model.type} with dims_in = {dims_in}, and dims_out = {dims_out}"
    )

    if cfg.model.type == "LGATr":
        model = eval(cfg.model.type)(
            logger=logger,
            process=cfg.dataset.process,
            cfg=cfg.model,
            dims_in=dims_in,
            dims_out=dims_out,
        ).to(device)
    else:
        model = eval(cfg.model.type)(
            logger=logger,
            process=cfg.dataset.process,
            cfg=cfg.model,
            dims_in=dims_in,
            dims_out=dims_out,
        ).to(device)
    
    model.name = cfg.model.type
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    model.init_dataloaders(dataset)

    if cfg.run.type == "train":
        # TRAIN MODEL
        model.train()
        os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
        model_path = os.path.join(run_dir, "model", "model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model to " + model_path)
    elif cfg.run.type == "plot":
        # LOAD MODEL
        model_path = os.path.join(run_dir, "model", "model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Loaded model from " + model_path)

    ### EVALUATE MODEL ###
    dataset.predicted_factors_ppd = {}
    for k in ["trn", "tst", "val"]:
        dataset.predicted_factors_ppd[k] = model.evaluate(
            loader=getattr(model, f"{k}loader")
        )
    dataset.apply_preprocessing(reverse=True)
    # ### COMPUTE OBSERVABLES ###
    logger.info("Computing observables")
    compute_observables(dataset, split="tst", include_ppd=True)

    ### MAKE PLOTS ###
    logger.info(f"Starting plots")
    plots = Plots(
        logger,
        dataset,
        model.losses if hasattr(model, "losses") else None,
        process_name=cfg.dataset.process,
        debug=False,
        model_name=model.name,
    )

    if hasattr(model, "losses"):
        logger.info(f"    Plotting train metrics")
        plots.plot_train_metrics(os.path.join(run_dir, f"train_metrics.pdf"), logy=True)
    logger.info(f"    Plotting ratio correlation")
    plots.plot_ratio_correlation(os.path.join(run_dir, f"ratio_corr_tst.pdf"), split="tst")
    logger.info(f"    Plotting regressed factors")
    plots.plot_weights(os.path.join(run_dir, f"factor_tst.pdf"), split="tst")
    plots.plot_weights(os.path.join(run_dir, f"factor_trn.pdf"), split="trn")
    plots.plot_weights(os.path.join(run_dir, f"factor_val.pdf"), split="val")
    logger.info(f"    Plotting ppd regressed factors")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_trn.pdf"), split="trn")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_tst.pdf"), split="tst")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_val.pdf"), split="val")
    logger.info(f"    Plotting observables")
    plots.plot_observables(os.path.join(run_dir, f"observables.pdf"))
    logger.info(f"    Plotting ppd observables")
    plots.plot_observables_ppd(os.path.join(run_dir, f"observables_ppd.pdf"))

    if device == torch.device("cuda"):
        max_used = torch.cuda.max_memory_allocated()
        max_total = torch.cuda.mem_get_info()[1]
        logger.info(
            f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
        )
    logger.info("Run finished")
