import os
import glob
import argparse
import yaml
from datetime import datetime
from madrecolor.utils.logger import setup_logging
from madrecolor.datasets.gluons import gg_ng
from madrecolor.datasets.dataset import compute_observables
from madrecolor.models import Model, MLP, MDN
from madrecolor.plots import Plots
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


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    parser.add_argument("path")
    return parser.parse_args()


def main():
    """
    Main function. Parse arguments, initialize directories and load params, initialize logger, execute the run
    """
    args = parse_args()

    ### INITIALIZE DIRECTORIES AND LOAD PARAMS ###
    if args.type == "train":
        with open(args.path, "r") as f:
            params = yaml.safe_load(f)
        base_dir = os.path.dirname(os.path.realpath(__file__))
        results_dir = os.path.join(base_dir, "results")
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + params["run_name"]
        run_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "params.yaml"), "w") as f:
            yaml.dump(params, f)

    ### LOAD PARAMS FROM EXISTING RUN ###
    elif args.type == "plot":
        run_dir = args.path
        with open(os.path.join(run_dir, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)

    else:
        raise NotImplementedError(f"Argument {args.type} not recognised")

    ### INITIALIZE LOGGER ###
    logger = init_logger(run_dir)

    ### RUN ###
    try:
        run(logger, run_dir, params, args)
    except Exception as e:
        logger.error(e, exc_info=True)


def run(logger, run_dir, params, args):
    """
    Run training and/or plotting of the model with the given parameters.
    """

    ### INITALIZE RUN ###
    logger.info(f"Starting {args.type} run in {run_dir}")
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"Device {device}")

    # INITIALIZE DATASET AND PREPROCESSING ###
    logger.info(f"Dataset: {params['dataset_params']['process']}")
    logger.info(
        f"Using parameterisation {params['dataset_params']['parameterisation']}"
    )
    dataset = eval(params["dataset_params"]["type"])(params["dataset_params"])
    logger.info(
        f"    [Train, Test, Val] events: [{len(dataset.events['trn'])}, {len(dataset.events['tst'])}, {len(dataset.events['val'])}]"
    )
    dataset.apply_preprocessing()

    ### INITIALIZE MODEL AND DATALOADERS ###
    dims_out = (
        1
        if params["model_params"]["type"] == "MLP"
        else params["model_params"].get("num_components", 10)
    )  # predicting reweighting factor
    dims_in = len(dataset.channels) - 1
    logger.info(
        f"Building model {params['model_params']['type']} with dims_in = {dims_in}, and dims_out = {dims_out}"
    )
    model = eval(params["model_params"]["type"])(
        params["model_params"],
        logger,
        dims_in,
        dims_out,
    ).to(device)
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    model.init_dataloaders(dataset)

    if args.type == "train":
        ### TRAIN MODEL ###
        model.train()
        os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
        model_path = os.path.join(run_dir, "model", f"model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model to " + model_path)
    elif args.type == "plot":
        ### LOAD MODEL ###
        model_path = os.path.join(run_dir, "model", f"model.pth")
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
        process_name=params["dataset_params"]["process"],
        debug=False,
    )

    if hasattr(model, "losses"):
        logger.info(f"    Plotting train metrics")
        plots.plot_train_metrics(os.path.join(run_dir, f"train_metrics.pdf"), logy=True)
    logger.info(f"    Plotting observables")
    plots.plot_observables(os.path.join(run_dir, f"observables.pdf"))
    logger.info(f"    Plotting ppd observables")
    plots.plot_observables_ppd(os.path.join(run_dir, f"observables_ppd.pdf"))
    logger.info(f"    Plotting regressed factors")
    plots.plot_weights(os.path.join(run_dir, f"factor_trn.pdf"), split="trn")
    plots.plot_weights(os.path.join(run_dir, f"factor_tst.pdf"), split="tst")
    plots.plot_weights(os.path.join(run_dir, f"factor_val.pdf"), split="val")
    logger.info(f"    Plotting ppd regressed factors")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_trn.pdf"), split="trn")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_tst.pdf"), split="tst")
    plots.plot_weights_ppd(os.path.join(run_dir, f"factor_ppd_val.pdf"), split="val")

    if device == torch.device("cuda"):
        max_used = torch.cuda.max_memory_allocated()
        max_total = torch.cuda.mem_get_info()[1]
        logger.info(
            f"GPU RAM information: max_used = {max_used/1e9:.3} GB, max_total = {max_total/1e9:.3} GB"
        )
    logger.info("Run finished")
