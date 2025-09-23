import warnings
import pickle
import os
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from src.datasets.dataset import Observable
from dataclasses import dataclass
from matplotlib.ticker import ScalarFormatter
from src.utils.data import unw_eff1, eval_time


@dataclass
class Metric:
    name: str
    value: float
    format: Optional[str] = None
    unit: Optional[str] = None
    tex_label: Optional[str] = None


@dataclass
class Line:
    y: np.ndarray
    y_err: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: Optional[str] = "solid"
    fill: bool = False
    vline: bool = False
    alpha: float = 1.0
    linewidth: float = 1.0


def optimize_gain_factor_2(
    process,
    model_name,
    pred,
    truth,
    t_LC,
    eff_LC,
    t_FC,
    t_surr,
    n_hel=None,
    alpha_min=0.8,
    frac_ow_max=0.001,
    grid_elements=201,
):
    """
    Optimize gain factor for a given process and model.
    Args:
        process (str): Process name.
        model_name (str): Model name.
        pred (np.ndarray): Predicted reweight factors (or R in case of the R method).
        truth (np.ndarray): True reweight factors (ALSO for the R method; this does not change).
        t_LC (float): Time for LC.
        eff_LC (float): Efficiency for LC.
        t_FC (float): Time for FC.
        t_surr (float): Time for surrogate model evaluation. Automatically multiplied by n_hel if n_hel is not None (in R method)
        n_hel (int, optional): Number of helicities. Defaults to None.
        alpha_min (float, optional): Minimum alpha value. Defaults to 0.8.
        grid_elements (int, optional): Number of grid elements. Defaults to 201.
    Returns:
        dict: Dictionary containing the optimized gain factor and other metrics.
    """

    t_surr = t_surr * n_hel if n_hel is not None else t_surr
    pcts = np.linspace(0.0, 100.0, grid_elements)

    pred = np.asarray(pred, float)
    ratio = np.asarray(truth, float) / pred
    eff_r = np.mean(truth) / np.max(truth)
    N = pred.size

    perc_pred = {p: np.percentile(pred, p) for p in pcts}
    perc_ratio = {p: np.percentile(ratio, p) for p in pcts}

    best_gain, best = -np.inf, None
    num = (1 / eff_r) * (t_LC * (1 / eff_LC) + t_FC)

    for p1 in pcts:
        e1 = np.mean(pred) / perc_pred[p1]
        if e1 > 1.0:
            continue
        ow1 = pred / perc_pred[p1]
        for p2 in pcts:
            e2 = np.mean(ratio) / perc_ratio[p2]
            if e2 > 1.0:
                continue
            ow2 = ratio / perc_ratio[p2]

            w = np.maximum(1.0, ow1 * ow2)
            frac_ow = float((w > 1).sum()) / N
            if frac_ow_max is not None and frac_ow > frac_ow_max:
                continue

            N_eff = (np.sum(w)) ** 2 / np.sum(w**2)
            alpha = N_eff / N

            if alpha < alpha_min:
                continue

            den_inner = (1 / e2) * ((1 / eff_LC) * (t_surr / e1 + t_LC) + t_FC)
            gain = alpha * num / den_inner  # == gain_factor2(...)
            if gain > best_gain:

                best_gain = gain
                best = dict(
                    gain=gain,
                    p1=p1,
                    p2=p2,
                    eff1=e1,
                    eff2=e2,
                    alpha=alpha,
                    n_ow=float((w > 1).sum()),
                    frac_ow=frac_ow,
                    mean_ow=np.nanmean(w[w > 1]),
                    mean_w=np.nanmean(w),
                    eff_LC=eff_LC,
                    t_LC=t_LC,
                    t_FC=t_FC,
                    t_surr=t_surr,
                    n_hel=n_hel,
                    eff_r=eff_r,
                )
    return best


def optimize_gain_factor_1(
    process,
    model_name,
    pred,
    truth,
    t_LC,
    eff_LC,
    t_FC,
    t_surr,
    alpha_min=0.8,
    frac_ow_max=0.001,
    grid_elements=241,
):
    """
    Optimize gain factor for a given process and model.
    Args:
        process (str): Process name.
        model_name (str): Model name.
        pred (np.ndarray): Predicted reweight factors (or R in case of the R method).
        truth (np.ndarray): True reweight factors (ALSO for the R method; this does not change).
        t_LC (float): Time for LC.
        eff_LC (float): Efficiency for LC.
        t_FC (float): Time for FC.
        t_surr (float): Time for surrogate model evaluation. Automatically multiplied by n_hel if n_hel is not None (in R method)
        n_hel (int, optional): Number of helicities. Defaults to None.
        alpha_min (float, optional): Minimum alpha value. Defaults to 0.8.
        grid_elements (int, optional): Number of grid elements. Defaults to 201.
    Returns:
        dict: Dictionary containing the optimized gain factor and other metrics.
    """
    pcts = np.linspace(40.00, 100.0, grid_elements)
    if frac_ow_max is not None:
        print("Allowed fraction of overweight events:", frac_ow_max)

    print("Sweep over percentages: ...", pcts[-10:])

    pred = np.asarray(pred, float)
    ratio = np.asarray(truth, float) / pred
    eff_r = np.mean(truth) / np.max(truth)
    N = pred.size

    perc_pred = {p: np.percentile(pred, p) for p in pcts}
    perc_ratio = {p: np.percentile(ratio, p) for p in pcts}

    best_gain, best = -np.inf, None
    num = (1 / eff_r) * (t_LC * (1 / eff_LC) + t_FC)

    for p1 in pcts:
        e1 = np.mean(pred) / perc_pred[p1]
        if e1 > 1.0:
            continue
        ow1 = pred / perc_pred[p1]
        for p2 in pcts:
            e2 = np.mean(ratio) / perc_ratio[p2]
            if e2 > 1.0:
                continue
            ow2 = ratio / perc_ratio[p2]

            w = np.maximum(1.0, ow2 * np.maximum(1.0, ow1))

            frac_ow = float((w > 1).sum()) / N
            if frac_ow_max is not None and frac_ow > frac_ow_max:
                continue

            N_eff = (np.sum(w)) ** 2 / np.sum(w**2)
            alpha = N_eff / N

            if alpha < alpha_min:
                continue

            den_inner = (1 / e2) * ((1 / e1) * (t_LC / eff_LC + t_surr) + t_FC)
            gain = alpha * num / den_inner  # == gain_factor2(...)
            if gain > best_gain:

                best_gain = gain
                best = dict(
                    gain=gain,
                    p1=p1,
                    p2=p2,
                    eff1=e1,
                    eff2=e2,
                    alpha=alpha,
                    n_ow=float((w > 1).sum()),
                    frac_ow=frac_ow,
                    mean_ow=np.nanmean(w[w > 1]),
                    mean_w=np.nanmean(w),
                    eff_LC=eff_LC,
                    t_LC=t_LC,
                    t_FC=t_FC,
                    t_surr=t_surr,
                    eff_r=eff_r,
                )

    return best


def compute_and_log_metrics(
    process: str,
    model_name: str,
    reweight_factors_pred: np.ndarray,
    reweight_factors_truth: np.ndarray,
    R: Optional[np.ndarray],
    n_helicities: int,
    split: str,
    ppd: bool,
    metrics: dict,
    log_file: str | None = None,
    file: str | None = None,
):
    """Compute metrics, update dictionary in-place, and append to log file."""

    ratio = reweight_factors_truth / reweight_factors_pred
    ratio_R = reweight_factors_truth / R if R is not None else None
    delta = (reweight_factors_pred - reweight_factors_truth) / reweight_factors_truth
    abs_delta = np.abs(delta)

    if process in eval_time and process in unw_eff1 and model_name in eval_time[process]:
        opt_dict_algo1 = optimize_gain_factor_1(
            process=process,
            model_name=model_name,
            pred=reweight_factors_pred,
            truth=reweight_factors_truth,
            t_LC=eval_time[process]["LC"],
            eff_LC=unw_eff1[process]["LC"],
            t_FC=eval_time[process]["FC"],
            t_surr=eval_time[process][model_name]["t_eval"],
            alpha_min=0.995,
            frac_ow_max=None,
            grid_elements=241 if split == "tst" and not ppd else 21,
        )
        opt_dict_algo1_p1max = opt_dict_algo1["p1"]
        opt_dict_algo1_p2max = opt_dict_algo1["p2"]

        print("\n")
        print("Result from optimizing algo1:", opt_dict_algo1)
        if R is not None:
            opt_dict_algo2 = optimize_gain_factor_2(
                process=process,
                model_name=model_name,
                pred=R,
                n_hel=n_helicities,
                truth=reweight_factors_truth,
                t_LC=eval_time[process]["LC"],
                eff_LC=unw_eff1[process]["LC"],
                t_FC=eval_time[process]["FC"],
                t_surr=eval_time[process][model_name]["t_eval"],
                alpha_min=0.8,
                frac_ow_max=0.001,
                grid_elements=201 if split == "tst" and not ppd else 21,
            )
            opt_dict_algo2_p1max = opt_dict_algo2["p1"]
            opt_dict_algo2_p2max = opt_dict_algo2["p2"]

            print("\n")
            print("Result from optimizing algo2:", opt_dict_algo2)

        metrics_update_algo1 = {
            "eff_1st_surr_opt_algo1": Metric(
                name="eff_1st_surr_opt_algo1",
                value=np.mean(reweight_factors_pred)
                / np.percentile(reweight_factors_pred, opt_dict_algo1_p1max),
                unit="",
                format="{:.5f}",
                tex_label=rf"\epsilon_{{\text{{1st, algo1}}}}^{{{opt_dict_algo1_p1max}}}",
            ),
            "eff_2nd_std": Metric(
                name="eff_2nd_std",
                value=np.mean(reweight_factors_truth) / np.max(reweight_factors_truth),
                unit="",
                format="{:.3f}",
                tex_label=r"\epsilon_{\text{2nd, std}}",
            ),
            "eff_2nd_surr_opt_algo1": Metric(
                name="eff_2nd_surr_opt_algo1",
                value=np.mean(ratio) / np.percentile(ratio, opt_dict_algo1_p2max),
                unit="",
                format="{:.5f}",
                tex_label=rf"\epsilon_{{\text{{2nd, algo1}}}}^{{{opt_dict_algo1_p2max}}}",
            ),
            "gain_algo1": Metric(
                name="gain_algo1",
                value=opt_dict_algo1["gain"],
                unit="",
                format="{:.3f}",
                tex_label=r"f^{\text{eff}}_{\text{algo1}}",
            ),
            "alpha_algo1": Metric(
                name="alpha_algo1",
                value=opt_dict_algo1["alpha"],
                unit="",
                format="{:.3f}",
                tex_label=r"\alpha^{\text{eff}}_{\text{algo1}}",
            ),
            "n_ow_algo1": Metric(
                name="n_ow_algo1",
                value=opt_dict_algo1["n_ow"],
                unit="",
                format="{:.3f}",
                tex_label=r"n_{\text{ow, algo1}}",
            ),
            "frac_ow_algo1": Metric(
                name="frac_ow_algo1",
                value=opt_dict_algo1["frac_ow"],
                unit="",
                format="{:.3f}",
                tex_label=r"f_{\text{ow, algo1}}",
            ),
            "mean_ow_algo1": Metric(
                name="mean_ow_algo1",
                value=opt_dict_algo1["mean_ow"],
                unit="",
                format="{:.3f}",
                tex_label=r"\langle w \rangle_{\text{algo1}}",
            ),
            "mean_w_algo1": Metric(
                name="mean_w_algo1",
                value=opt_dict_algo1["mean_w"],
                unit="",
                format="{:.3f}",
                tex_label=r"\langle w \rangle_{\text{algo1}}",
            ),
            "t_surr": Metric(
                name="t_surr",
                value=eval_time[process][model_name]["t_eval"],
                unit="s",
                format="{:.2f}",
                tex_label=r"t_{\text{surr}}",
            ),
            # "eff_1st_surr_pm9995": Metric(
            #     name="eff_1st_surr_pm9995",
            #     value=np.mean(reweight_factors_pred) / np.percentile(reweight_factors_pred, 99.95),
            #     unit="",
            #     format="{:.3f}",
            #     tex_label=r"\epsilon_{\text{1st, surr, 99.95}}",
            # ),
            # "eff_1st_surr_pm999": Metric(
            #     name="eff_1st_surr_pm999",
            #     value=np.mean(reweight_factors_pred) / np.percentile(reweight_factors_pred, 99.9),
            #     unit="",
            #     format="{:.3f}",
            #     tex_label=r"\epsilon_{\text{1st, surr, 99.9}}",
            # ),
            # "eff_1st_surr_pm995": Metric(
            #     name="eff_1st_surr_pm995",
            #     value=np.mean(reweight_factors_pred) / np.percentile(reweight_factors_pred, 99.5),
            #     unit="",
            #     format="{:.3f}",
            #     tex_label=r"\epsilon_{\text{1st, surr, 99.5}}",
            # ),
            # "eff_2nd_surr_pm9999": Metric(
            #     name="eff_2nd_surr_pm9999",
            #     value=np.mean(ratio) / np.percentile(ratio, 99.99),
            #     unit="",
            #     format="{:.2f}",
            #     tex_label=r"\epsilon_{99.99}",
            # ),
            # "eff_2nd_surr_pm9995": Metric(
            #     name="eff_2nd_surr_pm9995",
            #     value=np.mean(ratio) / np.percentile(ratio, 99.95),
            #     unit="",
            #     format="{:.2f}",
            #     tex_label=r"\epsilon_{99.95}",
            # ),
            # "eff_2nd_surr_pm995": Metric(
            #     name="eff_2nd_surr_pm995",
            #     value=np.mean(ratio) / np.percentile(ratio, 99.5),
            #     unit="",
            #     format="{:.2f}",
            #     tex_label=r"\epsilon_{99.5}",
            # ),
        }
        metrics.update(metrics_update_algo1)
        if R is not None:
            metrics_update_algo2 = {
                "eff_1st_surr_opt_algo2": Metric(
                    name="eff_1st_surr_opt_algo2",
                    value=np.mean(R) / np.percentile(R, opt_dict_algo2_p1max),
                    unit="",
                    format="{:.5f}",
                    tex_label=rf"\epsilon_{{\text{{1st, algo2}}}}^{{{opt_dict_algo2_p1max}}}",
                ),
                "eff_2nd_surr_opt_algo2": Metric(
                    name="eff_2nd_surr_opt_algo2",
                    value=np.mean(ratio_R) / np.percentile(ratio_R, opt_dict_algo2_p2max),
                    unit="",
                    format="{:.5f}",
                    tex_label=rf"\epsilon_{{\text{{2nd, algo2}}}}^{{{opt_dict_algo2_p2max}}}",
                ),
                "gain_algo2": Metric(
                    name="gain_algo2",
                    value=opt_dict_algo2["gain"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"f^{\text{eff}}_{\text{algo2}}",
                ),
                "alpha_algo2": Metric(
                    name="alpha_algo2",
                    value=opt_dict_algo2["alpha"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"\alpha^{\text{eff}}_{\text{algo2}}",
                ),
                "n_ow_algo2": Metric(
                    name="n_ow_algo2",
                    value=opt_dict_algo2["n_ow"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"n_{\text{ow, algo2}}",
                ),
                "frac_ow_algo2": Metric(
                    name="frac_ow_algo2",
                    value=opt_dict_algo2["frac_ow"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"f_{\text{ow, algo2}}",
                ),
                "mean_ow_algo2": Metric(
                    name="mean_ow_algo2",
                    value=opt_dict_algo2["mean_ow"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"\langle w \rangle_{\text{algo2}}",
                ),
                "mean_w_algo2": Metric(
                    name="mean_w_algo2",
                    value=opt_dict_algo2["mean_w"],
                    unit="",
                    format="{:.3f}",
                    tex_label=r"\langle w \rangle_{\text{algo2}}",
                ),
                "n_helicities": Metric(
                    name="n_helicities",
                    value=n_helicities,
                    unit="",
                    format="{:.0f}",
                    tex_label=r"n_{\text{hel}}",
                ),
            }
            metrics.update(metrics_update_algo2)
        # Append to log file
        if log_file is None:
            if file is None:
                raise ValueError("Need either log_file or pdf_file for metrics storage.")
            log_file = os.path.join(os.path.dirname(file) or ".", "metrics.log")
        with open(log_file, "a") as f:
            f.write(f"split_{split}-ppd_{ppd}: ")
            for i, (k, m) in enumerate(metrics.items()):
                value = f"{m.value:.10f}" if np.abs(m.value) > 1e-7 else f"{m.value:.2e}"
                f.write(f"{k}: {value}")
                if i < len(metrics) - 1:
                    f.write(", ")
            f.write("\n")
            f.close()
    else:
        base_metrics = {
            "eff_1st_surr_algo1": Metric(
                name="eff_1st_surr_algo1",
                value=np.mean(reweight_factors_pred) / np.max(reweight_factors_pred),
                unit="",
                format="{:.3f}",
                tex_label=r"\epsilon_{\text{1st, surr}}",
            ),
            "eff_2nd_std": Metric(
                name="eff_2nd_std",
                value=np.mean(reweight_factors_truth) / np.max(reweight_factors_truth),
                unit="",
                format="{:.3f}",
                tex_label=r"\epsilon_{\text{2nd, std}}",
            ),
            "eff_2nd_surr_algo1": Metric(
                name="eff_2nd_surr_algo1",
                value=np.mean(ratio) / np.max(ratio),
                unit="",
                format="{:.2f}",
                tex_label=r"\epsilon_{100}",
            ),
            "t_LC": Metric(
                name="t_LC",
                value=eval_time[process]["LC"],
                unit="s",
                format="{:.2f}",
                tex_label=r"t_{\text{LC}}",
            ),
            "eff_LC": Metric(
                name="eff_LC",
                value=unw_eff1[process]["LC"],
                unit="",
                format="{:.3f}",
                tex_label=r"\epsilon_{\text{LC}}",
            ),
            "t_FC": Metric(
                name="t_FC",
                value=eval_time[process]["FC"],
                unit="s",
                format="{:.2f}",
                tex_label=r"t_{\text{FC}}",
            ),
            "ratio_mean": Metric(
                name="ratio_mean",
                value=np.mean(ratio),
                unit="",
                tex_label=r"\mu",
            ),
            "ratio_std": Metric(
                name="ratio_std",
                value=np.std(ratio),
                unit="",
                tex_label=r"\sigma",
            ),
            "ratio_max": Metric(
                name="ratio_max",
                value=np.max(ratio),
                unit="",
                tex_label=r"x_{\max}",
            ),
            "delta_mean": Metric(
                name="delta_mean",
                value=np.mean(delta),
                unit="",
                tex_label=r"\mu",
            ),
            "delta_std": Metric(
                name="delta_std",
                value=np.std(delta),
                unit="",
                tex_label=r"\sigma",
            ),
            "abs_delta_mean": Metric(
                name="abs_delta_mean",
                value=np.mean(abs_delta),
                unit="",
                tex_label=r"\mu",
            ),
            "abs_delta_std": Metric(
                name="abs_delta_std",
                value=np.std(abs_delta),
                unit="",
                tex_label=r"\sigma",
            ),
            "abs_delta_qmin": Metric(
                name="abs_delta_qmin",
                value=np.percentile(abs_delta, 0.05),
                unit="",
                tex_label=r"q_{0.05}",
            ),
            "abs_delta_qmax": Metric(
                name="abs_delta_qmax",
                value=np.percentile(abs_delta, 99.95),
                unit="",
                tex_label=r"q_{99.95}",
            ),
            "abs_delta_min": Metric(
                name="abs_delta_min",
                value=np.min(abs_delta),
                unit="",
                tex_label=r"\min",
            ),
            "abs_delta_max": Metric(
                name="abs_delta_max",
                value=np.max(abs_delta),
                unit="",
                tex_label=r"\max",
            ),
        }
        if R is not None:
            base_metrics["eff_1st_surr_algo2"] = Metric(
                name="eff_1st_surr_algo2",
                value=np.mean(R) / np.max(R),
                unit="",
                format="{:.3f}",
                tex_label=r"\epsilon_{\text{1st, R}}",
            )
            base_metrics["eff_2nd_surr_algo2"] = Metric(
                name="eff_2nd_surr_algo2",
                value=np.mean(ratio_R) / np.max(ratio_R),
                unit="",
                format="{:.2f}",
                tex_label=r"\epsilon_{\text{2nd, R}}",
            )
        metrics.update(base_metrics)
        if log_file is None:
            if file is None:
                raise ValueError("Need either log_file or pdf_file for metrics storage.")
            log_file = os.path.join(os.path.dirname(file) or ".", "metrics.log")
        with open(log_file, "a") as f:
            f.write(f"split_{split}-ppd_{ppd}: ")
            for i, (k, m) in enumerate(metrics.items()):
                value = f"{m.value:.10f}" if np.abs(m.value) > 1e-7 else f"{m.value:.2e}"
                f.write(f"{k}: {value}")
                if i < len(metrics) - 1:
                    f.write(", ")
            f.write("\n")
            f.close()


def hist_weights_plot(
    pdf: PdfPages,
    lines: list[Line],
    bins: np.ndarray,
    show_ratios: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: str = f"$r(x)$",
    ylabel: Optional[str] = None,
    no_scale: bool = False,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    metrics: Optional[float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    model_name: Optional[str] = "NN",
    rect=(0.13, 0.18, 0.96, 0.96),
    size_multipler: float = 1.0,
    legend_kwargs: Optional[dict] = None,
):
    """
    Makes a single histogram plot for the weights
    Args:
        pdf: Multipage PDF object
        lines: List of line objects describing the histograms
        bins: Numpy array with the bin boundaries
        show_ratios: If True, show a panel with ratios
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        n_panels = 1 + int(show_ratios) + int(metrics is not None)
        fig, axs = plt.subplots(
            n_panels,
            1,
            sharex=True,
            figsize=(size_multipler * 6, size_multipler * 4.5),
            gridspec_kw={
                "height_ratios": (12, 1 + 2 * int(show_ratios), 1)[:n_panels],
                "hspace": 0.00,
            },
        )

        if n_panels == 1:
            axs = [axs]

        for line in lines:
            if line.vline:
                axs[0].axvline(
                    line.y,
                    label=line.label,
                    color=line.color,
                    linestyle=line.linestyle,
                )
                if line.y_err is not None:
                    # plot a filling between -y_err and +y_err
                    axs[0].axvspan(
                        line.y - line.y_err,
                        line.y + line.y_err,
                        color=line.color,
                        alpha=0.15,
                        label=None,
                        linestyle="solid",
                        linewidth=0.5 * line.linewidth,
                    )
                continue
            integral = np.sum((bins[1:] - bins[:-1]) * line.y)
            scale = 1 / integral if integral != 0.0 else 1.0
            if line.y_ref is not None:
                ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
            if no_scale:
                scale = 1.0
                ref_scale = 1.0

            hist_line(
                axs[0],
                bins,
                line.y * scale,
                line.y_err * scale if line.y_err is not None else None,
                label=line.label,
                color=line.color,
                fill=line.fill,
                linestyle=line.linestyle,
                alpha=line.alpha,
            )
            axs[0].set_xlim(bins[0], bins[-1]) if xlim is None else axs[0].set_xlim(*xlim)
            if line.y_ref is not None and show_ratios:
                ratio = (line.y * scale) / (line.y_ref * ref_scale)
                ratio_isnan = np.isnan(ratio)
                if line.y_err is not None:
                    if len(line.y_err.shape) == 2:
                        ratio_err = (line.y_err * scale) / (line.y_ref * ref_scale)
                        ratio_err[:, ratio_isnan] = 0.0
                    else:
                        ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                        ratio_err[ratio_isnan] = 0.0
                else:
                    ratio_err = None
                ratio[ratio_isnan] = 1.0
                hist_line(
                    axs[1],
                    bins,
                    ratio,
                    ratio_err,
                    label=None,
                    color=line.color,
                    alpha=line.alpha,
                    linestyle=line.linestyle,
                )
        if show_ratios:
            axs[1].set_ylabel(f"$\\frac{{\\text{{{model_name}}}}}{{\\text{{Truth}}}}$")
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].set_xlim(bins[0], bins[-1]) if xlim is None else axs[1].set_xlim(*xlim)
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

        if metrics is not None:
            for i, metric_type in enumerate(metrics.keys()):
                metric = metrics[metric_type]
                formatted_value = (
                    metric.format.format(metric.value)
                    if metric.format is not None
                    else f"{metric.value:.3f}"
                    if np.abs(metric.value) > 1e-2
                    else f"{metric.value:.3e}"
                )
                label = (
                    f"${metric.tex_label}$"
                    if metric.tex_label is not None
                    else f"{metric.name}"
                )
                metric_str = rf"{label}$ = ${formatted_value}" + (
                    rf"$\ \mathrm{{{metric.unit}}}$" if metric.unit else ""
                )
                axs[-1].text(
                    0.025 + (1.01 / len(metrics)) * i,
                    0.1,
                    metric_str,
                    fontsize=10,
                    transform=axs[-1].transAxes,
                    va="bottom",
                )
            axs[-1].set_yticks([])

        if title is not None:
            corner_text(axs[0], title, "left", "top")
            if subtitle is not None:
                corner_text(axs[0], subtitle, "left", "top", is_subtitle=True)
        if legend_kwargs is None:
            axs[0].legend(loc="best", frameon=False, handlelength=1.0)
        else:
            legend_kwargs.pop("handlelength", 1.0)
            axs[0].legend(frameon=False, handlelength=1.0, **legend_kwargs)
        axs[0].set_ylabel("Normalized") if not no_scale and ylabel is None else axs[
            0
        ].set_ylabel("Events") if ylabel is None else axs[0].set_ylabel(ylabel)
        axs[0].set_xscale("linear" if xscale is None else xscale)
        yscale = "log" if yscale is None else yscale
        axs[0].set_yscale(yscale)
        if ylim is not None:
            axs[0].set_ylim(*ylim)

        axs[-1].set_xlabel(xlabel)
        axs[-1].set_xscale("linear" if xscale is None else xscale)

        fig.subplots_adjust(left=rect[0], bottom=rect[1], right=rect[2], top=rect[3])
        plt.savefig(pdf, format="pdf")
        plt.close()


def append_to_pickle(pickle_file, new_entry):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        pickle_data = []

    pickle_data.append(new_entry)
    with open(pickle_file, "wb") as f:
        pickle.dump(pickle_data, f)


def compute_hist_data(bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
    if bayesian:
        hists = np.stack(
            [np.histogram(d, bins=bins, density=False, weights=weights)[0] for d in data],
            axis=0,
        )
        y = hists[0]
        y_err = np.std(hists, axis=0)
    else:
        y, _ = np.histogram(data, bins=bins, density=False, weights=weights)
        y_err = np.sqrt(y)
    return y, y_err


def hist_plot(
    pdf: PdfPages,
    lines: list[Line],
    bins: np.ndarray,
    observable: Observable,
    show_ratios: bool = True,
    title: Optional[str] = None,
    legend_kwargs: Optional[dict] = None,
    no_scale: bool = False,
    yscale: Optional[str] = None,
    debug=False,
    model_name: str = "NN",
):
    """
    Makes a single histogram plot, used for the observable histograms and clustering
    histograms.
    Args:
        pdf: Multipage PDF object
        lines: List of line objects describing the histograms
        bins: Numpy array with the bin boundaries
        show_ratios: If True, show a panel with ratios
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        n_panels = 1 + int(show_ratios)
        fig, axs = plt.subplots(
            n_panels,
            1,
            sharex=True,
            figsize=(6, 4.5),
            gridspec_kw={"height_ratios": (12, 3, 1)[:n_panels], "hspace": 0.00},
        )
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.08, 0.1, 0.97, 0.96))
        if n_panels == 1:
            axs = [axs]

        for line in lines:
            if line.vline:
                axs[0].axvline(
                    line.y, label=line.label, color=line.color, linestyle=line.linestyle
                )
                continue
            integral = np.sum((bins[1:] - bins[:-1]) * line.y)
            scale = 1 / integral if integral != 0.0 else 1.0
            if line.y_ref is not None:
                ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
            if no_scale:
                scale = 1.0
                ref_scale = 1.0

            if debug:
                print("Actual values plotted:", line.y * scale)
            hist_line(
                axs[0],
                bins,
                line.y * scale,
                line.y_err * scale if line.y_err is not None else None,
                label=line.label,
                color=line.color,
                fill=line.fill,
                linestyle=line.linestyle,
                linewidth=line.linewidth,
            )

            if line.y_ref is not None:
                ratio = (line.y * scale) / (line.y_ref * ref_scale)
                ratio_isnan = np.isnan(ratio)
                if line.y_err is not None:
                    if len(line.y_err.shape) == 2:
                        ratio_err = (line.y_err * scale) / (line.y_ref * ref_scale)
                        ratio_err[:, ratio_isnan] = 0.0
                    else:
                        ratio_err = np.sqrt((line.y_err / line.y) ** 2)
                        ratio_err[ratio_isnan] = 0.0
                else:
                    ratio_err = None
                ratio[ratio_isnan] = 1.0
                hist_line(
                    axs[1],
                    bins,
                    ratio,
                    ratio_err,
                    label=None,
                    color=line.color,
                    linestyle=line.linestyle,
                    linewidth=line.linewidth,
                )

        if title is not None:
            corner_text(axs[0], title, "left", "top")
        axs[0].legend(
            frameon=False, **(legend_kwargs if legend_kwargs is not None else {})
        )
        axs[0].set_ylabel("Normalized")
        axs[0].set_yscale(observable.yscale if yscale is None else yscale)
        axs[0].set_xscale(
            observable.xscale if observable.xscale is not None else "linear"
        )

        if show_ratios:
            axs[1].set_ylabel(f"$\\frac{{\\text{{{model_name}}}}}{{\\text{{Truth}}}}$")
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)
            axs[1].set_xscale(observable.xscale)

        unit = "" if observable.unit is None else f"[{observable.unit}]"
        axs[-1].set_xlabel(f"${{{observable.tex_label}}}$ $\ {unit}$")
        axs[-1].set_xlim(bins[0], bins[-1])
        plt.savefig(pdf, format="pdf")
        plt.close()


def safe_lognorm(data: np.ndarray) -> LogNorm:
    """
    Create a safe norm for the data.
    Args:
        data: Data to normalize
    Returns:
        norm: Normalization object
    """
    flat = data[~np.isnan(data)]
    if flat.size > 0:
        # only positives make sense on log scale
        positives = flat[flat > 0]
        if positives.size > 0:
            vmin = positives.min()
            vmax = positives.max()
        else:
            # all data zero or NaN --> force a tiny range
            vmin, vmax = 1e-10, 1e-9
    else:
        # completely empty --> arbitrary fallback
        vmin, vmax = 1e-10, 1e-9
    return LogNorm(vmin=vmin, vmax=vmax)


def hist_line(
    ax: mpl.axes.Axes,
    bins: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    label: str,
    color: str,
    linestyle: str = "solid",
    fill: bool = False,
    alpha: float = 1.0,
    linewidth: float = 1.0,
):
    """
    Plot a stepped line for a histogram, optionally with error bars.
    Args:
        ax: Matplotlib Axes
        bins: Numpy array with bin boundaries
        y: Y values for the bins
        y_err: Y errors for the bins
        label: Label of the line
        color: Color of the line
        linestyle: line style
        fill: Filled histogram
    """

    dup_last = lambda a: np.append(a, a[-1])

    if fill:
        ax.fill_between(
            bins,
            dup_last(y),
            label=label,
            facecolor=color,
            step="post",
            alpha=0.2 * alpha,
        )
    else:
        ax.step(
            bins,
            dup_last(y),
            label=label,
            color=color,
            linewidth=linewidth,
            where="post",
            ls=linestyle,
            alpha=alpha,
        )
    if y_err is not None:
        if len(y_err.shape) == 2:
            y_low = y_err[0]
            y_high = y_err[1]
        else:
            y_low = y - y_err
            y_high = y + y_err

        ax.step(
            bins,
            dup_last(y_high),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5 * linewidth,
            where="post",
        )
        ax.step(
            bins,
            dup_last(y_low),
            color=color,
            alpha=0.5 * alpha,
            linewidth=0.5 * linewidth,
            where="post",
        )
        ax.fill_between(
            bins,
            dup_last(y_low),
            dup_last(y_high),
            facecolor=color,
            alpha=0.3 * alpha,
            step="post",
        )


def corner_text(
    ax: mpl.axes.Axes,
    text: str,
    horizontal_pos: str,
    vertical_pos: str,
    is_subtitle: bool = False,
):
    ax.text(
        x=0.95 if horizontal_pos == "right" else 0.05,
        y=0.95 - 0.1 * (is_subtitle)
        if vertical_pos == "top"
        else 0.05 + 0.1 * (is_subtitle),
        s=text,
        horizontalalignment=horizontal_pos,
        verticalalignment=vertical_pos,
        transform=ax.transAxes,
    )
    # Dummy line for automatic legend placement
    plt.plot(
        0.8 if horizontal_pos == "right" else 0.2,
        0.8 - 0.1 * (is_subtitle) if vertical_pos == "top" else 0.2 + 0.1 * (is_subtitle),
        transform=ax.transAxes,
        color="none",
    )


bins_dict = {
    "r": {
        "targets": {
            "gg_4g": np.linspace(0.85, 1.05, 64),
            "gg_5g": np.linspace(0.75, 1.15, 64),
            "gg_6g": np.linspace(0.65, 1.5, 64),
            "gg_7g": np.linspace(0.59, 2.0, 64),
            "gg_ddbar2g": np.linspace(0.35, 1.15, 64),
            "gg_ddbar3g": np.linspace(0.35, 1.15, 64),
            "gg_ddbar4g": np.linspace(0.35, 1.2, 64),
            "gg_ddbar5g": np.linspace(0.35, 1.5, 64),
            "dbard_4g": np.linspace(0.45, 0.9, 64),
            "dbard_5g": np.linspace(0.4, 1.0, 64),
            "dbard_6g": np.linspace(0.35, 1.15, 64),
            "dbard_7g": np.linspace(0.3, 1.2, 64),
            "gg_ddbaruubar0g_co1": np.linspace(0.2, 1.3, 64),
            "gg_ddbaruubar1g_co1": np.linspace(0.1, 1.3, 64),
            "gg_ddbaruubar2g_co1": np.linspace(0.05, 1.15, 64),
            "gg_ddbaruubar3g_co1": np.linspace(0.05, 1.15, 64),
            "gg_ddbaruubar0g_co2": np.linspace(0.2, 1.3, 64),
            "gg_ddbaruubar1g_co2": np.linspace(0.1, 1.3, 64),
            "gg_ddbaruubar2g_co2": np.linspace(0.05, 1.15, 64),
            "gg_ddbaruubar3g_co2": np.linspace(0.05, 1.15, 64),
            "ddbar_uubar2g_co1": np.linspace(0.3, 1.15, 64),
            "ddbar_uubar3g_co1": np.linspace(0.25, 1.15, 64),
            "ddbar_uubar4g_co1": np.linspace(0.05, 1.15, 64),
            "ddbar_uubar5g_co1": np.linspace(0.05, 1.15, 64),
            "ddbar_uubar2g_co2": np.linspace(0.3, 1.15, 64),
            "ddbar_uubar3g_co2": np.linspace(0.25, 1.15, 64),
            "ddbar_uubar4g_co2": np.linspace(0.05, 1.15, 64),
            "ddbar_uubar5g_co2": np.linspace(0.05, 1.15, 64),
        },
        "ratios": {
            "gg_4g": np.linspace(0.90, 1.1, 64),
            "gg_5g": np.linspace(0.90, 1.1, 64),
            "gg_6g": np.linspace(0.90, 1.1, 64),
            "gg_7g": np.linspace(0.90, 1.1, 64),
            "gg_ddbar2g": np.linspace(0.90, 1.1, 64),
            "gg_ddbar3g": np.linspace(0.90, 1.1, 64),
            "gg_ddbar4g": np.linspace(0.90, 1.1, 64),
            "gg_ddbar5g": np.linspace(0.90, 1.1, 64),
            "dbard_4g": np.linspace(0.90, 1.1, 64),
            "dbard_5g": np.linspace(0.90, 1.1, 64),
            "dbard_6g": np.linspace(0.90, 1.1, 64),
            "dbard_7g": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar0g_co1": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar1g_co1": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar2g_co1": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar3g_co1": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar0g_co2": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar1g_co2": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar2g_co2": np.linspace(0.90, 1.1, 64),
            "gg_ddbaruubar3g_co2": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar2g_co1": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar3g_co1": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar4g_co1": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar5g_co1": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar2g_co2": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar3g_co2": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar4g_co2": np.linspace(0.90, 1.1, 64),
            "ddbar_uubar5g_co2": np.linspace(0.90, 1.1, 64),
        },
        "deltas": {
            "gg_4g": np.linspace(-0.1, 0.1, 64),
            "gg_5g": np.linspace(-0.1, 0.1, 64),
            "gg_6g": np.linspace(-0.1, 0.1, 64),
            "gg_7g": np.linspace(-0.1, 0.1, 64),
            "gg_ddbar2g": np.linspace(-0.1, 0.1, 64),
            "gg_ddbar3g": np.linspace(-0.1, 0.1, 64),
            "gg_ddbar4g": np.linspace(-0.1, 0.1, 64),
            "gg_ddbar5g": np.linspace(-0.1, 0.1, 64),
            "dbard_4g": np.linspace(-0.1, 0.1, 64),
            "dbard_5g": np.linspace(-0.1, 0.1, 64),
            "dbard_6g": np.linspace(-0.1, 0.1, 64),
            "dbard_7g": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar0g_co1": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar1g_co1": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar2g_co1": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar3g_co1": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar0g_co2": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar1g_co2": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar2g_co2": np.linspace(-0.1, 0.1, 64),
            "gg_ddbaruubar3g_co2": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar2g_co1": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar3g_co1": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar4g_co1": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar5g_co1": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar2g_co2": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar3g_co2": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar4g_co2": np.linspace(-0.1, 0.1, 64),
            "ddbar_uubar5g_co2": np.linspace(-0.1, 0.1, 64),
        },
        "abs_deltas": {
            "gg_4g": np.logspace(-14, 2, 64),
            "gg_5g": np.logspace(-14, 2, 64),
            "gg_6g": np.logspace(-14, 2, 64),
            "gg_7g": np.logspace(-14, 2, 64),
            "gg_ddbar2g": np.logspace(-14, 2, 64),
            "gg_ddbar3g": np.logspace(-14, 2, 64),
            "gg_ddbar4g": np.logspace(-14, 2, 64),
            "gg_ddbar5g": np.logspace(-14, 2, 64),
            "dbard_4g": np.logspace(-14, 2, 64),
            "dbard_5g": np.logspace(-14, 2, 64),
            "dbard_6g": np.logspace(-14, 2, 64),
            "dbard_7g": np.logspace(-14, 2, 64),
            "gg_ddbaruubar0g_co1": np.logspace(-14, 2, 64),
            "gg_ddbaruubar1g_co1": np.logspace(-14, 2, 64),
            "gg_ddbaruubar2g_co1": np.logspace(-14, 2, 64),
            "gg_ddbaruubar3g_co1": np.logspace(-14, 2, 64),
            "gg_ddbaruubar0g_co2": np.logspace(-14, 2, 64),
            "gg_ddbaruubar1g_co2": np.logspace(-14, 2, 64),
            "gg_ddbaruubar2g_co2": np.logspace(-14, 2, 64),
            "gg_ddbaruubar3g_co2": np.logspace(-14, 2, 64),
            "ddbar_uubar2g_co1": np.logspace(-14, 2, 64),
            "ddbar_uubar3g_co1": np.logspace(-14, 2, 64),
            "ddbar_uubar4g_co1": np.logspace(-14, 2, 64),
            "ddbar_uubar5g_co1": np.logspace(-14, 2, 64),
            "ddbar_uubar2g_co2": np.logspace(-14, 2, 64),
            "ddbar_uubar3g_co2": np.logspace(-14, 2, 64),
            "ddbar_uubar4g_co2": np.logspace(-14, 2, 64),
            "ddbar_uubar5g_co2": np.logspace(-14, 2, 64),
        },
    },
    "FC": {
        "targets": {
            "gg_4g": np.logspace(-11, 5, 64),
            "gg_5g": np.logspace(-15, 3, 64),
            "gg_6g": np.logspace(-18, 2, 64),
            "gg_7g": np.logspace(-22, 1, 64),
        },
        "ratios": {
            "gg_4g": np.logspace(-3, 3, 64),
            "gg_5g": np.logspace(-3, 3, 64),
            "gg_6g": np.logspace(-3, 3, 64),
            "gg_7g": np.logspace(-3, 3, 64),
        },
        "deltas": {
            "gg_4g": np.linspace(-5, 30, 64),
            "gg_5g": np.linspace(-5, 30, 64),
            "gg_6g": np.linspace(-5, 30, 64),
            "gg_7g": np.linspace(-5, 30, 64),
        },
        "abs_deltas": {
            "gg_4g": np.logspace(-9, 4, 64),
            "gg_5g": np.logspace(-9, 4, 64),
            "gg_6g": np.logspace(-9, 4, 64),
            "gg_7g": np.logspace(-9, 4, 64),
        },
    },
    "LC": {
        "targets": {
            "gg_4g": np.logspace(-11, 5, 64),
            "gg_5g": np.logspace(-15, 3, 64),
            "gg_6g": np.logspace(-18, 2, 64),
            "gg_7g": np.logspace(-22, 1, 64),
        },
        "ratios": {
            "gg_4g": np.logspace(-3, 3, 64),
            "gg_5g": np.logspace(-3, 3, 64),
            "gg_6g": np.logspace(-3, 3, 64),
            "gg_7g": np.logspace(-3, 3, 64),
        },
        "deltas": {
            "gg_4g": np.linspace(-5, 30, 64),
            "gg_5g": np.linspace(-5, 30, 64),
            "gg_6g": np.linspace(-5, 30, 64),
            "gg_7g": np.linspace(-5, 30, 64),
        },
        "abs_deltas": {
            "gg_4g": np.logspace(-9, 4, 64),
            "gg_5g": np.logspace(-9, 4, 64),
            "gg_6g": np.logspace(-9, 4, 64),
            "gg_7g": np.logspace(-9, 4, 64),
        },
    },
}
