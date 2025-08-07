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
from collections import defaultdict

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


def compute_and_log_metrics(
    reweight_factors_pred: np.ndarray,
    reweight_factors_truth: np.ndarray,
    split: str,
    ppd: bool,
    metrics: dict,
    log_file: str | None = None,
    file: str | None = None,
):
    """Compute metrics, update dictionary in-place, and append to log file."""

    ratio = reweight_factors_truth / reweight_factors_pred
    delta = (reweight_factors_pred - reweight_factors_truth) / reweight_factors_truth
    abs_delta = np.abs(delta)

    metrics_update = {
        "r_pred_mean": Metric(
            name="r_pred_mean",
            value=np.mean(reweight_factors_pred),
            unit="",
            format="{:.3f}",
            tex_label=r"\overline{s}",
        ),
        "r_pred_max": Metric(
            name="r_pred_max",
            value=np.max(reweight_factors_pred),
            unit="",
            format="{:.3f}",
            tex_label=r"s_{\max}",
        ),
        "eff_1st_surr": Metric(
            name="eff_1st_surr",
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
            tex_label=r"\max",
        ),
        "eff_2nd_surr": Metric(
            name="eff_2nd_surr",
            value=np.mean(ratio) / np.max(ratio),
            unit="",
            format="{:.3f}",
            tex_label=r"\epsilon_{\text{2nd, surr}}",
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

    metrics.update(metrics_update)

    # Append to log file
    if log_file is None:
        if file is None:
            raise ValueError("Need either log_file or pdf_file for metrics storage.")
        log_file = os.path.join(os.path.dirname(file) or ".", "metrics.log")
    with open(log_file, "a") as f:
        f.write(f"split_{split}-ppd_{ppd}: ")
        for i, (k, m) in enumerate(metrics.items()):
            value = f"{m.value:.10f}" if np.abs(m.value) > 1e-7 else f"{m.value:.3e}"
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
    xlabel: str = f"$r(x)$",
    ylabel: Optional[str] = None,
    no_scale: bool = False,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    metrics: Optional[float] = None,
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
                        alpha=0.05,
                        label=None,
                        linestyle='solid',
                        linewidth=0.5*line.linewidth,
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
        axs[-1].set_xlim(bins[0], bins[-1])
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


def corner_text(ax: mpl.axes.Axes, text: str, horizontal_pos: str, vertical_pos: str):
    ax.text(
        x=0.95 if horizontal_pos == "right" else 0.05,
        y=0.95 if vertical_pos == "top" else 0.05,
        s=text,
        horizontalalignment=horizontal_pos,
        verticalalignment=vertical_pos,
        transform=ax.transAxes,
    )
    # Dummy line for automatic legend placement
    plt.plot(
        0.8 if horizontal_pos == "right" else 0.2,
        0.8 if vertical_pos == "top" else 0.2,
        transform=ax.transAxes,
        color="none",
    )
