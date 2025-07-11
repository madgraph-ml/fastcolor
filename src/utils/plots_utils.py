import warnings
import pickle
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from src.datasets.dataset import Observable
from dataclasses import dataclass
from matplotlib.ticker import ScalarFormatter

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

def hist_weights_plot(
    pdf: PdfPages,
    lines: list[Line],
    bins: np.ndarray,
    show_ratios: bool = False,
    title: Optional[str] = None,
    xlabel: str = f"$r(x)$",
    no_scale: bool = False,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    metrics: Optional[float] = None,
    ylim: tuple[float, float] = None,
    model_name: Optional[str] = "NN",
    rect=(0.13,0.18,0.96,0.96)
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
            figsize=(6, 4.5),
            gridspec_kw={"height_ratios": (12, 1+2*int(show_ratios), 1)[:n_panels], "hspace": 0.00},
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
                    axs[1], bins, ratio, ratio_err, label=None, color=line.color
                )
        if show_ratios:
            axs[1].set_ylabel(
                f"$\\frac{{\\text{{{model_name}}}}}{{\\text{{Truth}}}}$"
            )
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

        if metrics is not None:
            metrics = f"Loss = {metrics:.2e}" if np.abs(metrics) < 1e-4 else f"Loss = {metrics:.4f}"
            axs[-1].text(0.1, 0.1, metrics, fontsize=13, transform=axs[-1].transAxes, va='bottom')
            axs[-1].set_yticks([])

        if title is not None:
            corner_text(axs[0], title, "left", "top")
        axs[0].legend(loc="best", frameon=False)
        axs[0].set_ylabel("Normalized") if not no_scale else axs[0].set_ylabel("Events")
        axs[0].set_xscale("linear" if xscale is None else xscale)
        axs[0].set_yscale("log" if yscale is None else yscale)
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
                )

        if title is not None:
            corner_text(axs[0], title, "left", "top")
        axs[0].legend(
            frameon=False, **(legend_kwargs if legend_kwargs is not None else {})
        )
        axs[0].set_ylabel("Normalized")
        axs[0].set_yscale(observable.yscale if yscale is None else yscale)
        axs[0].set_xscale(observable.xscale if observable.xscale is not None else "linear")

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
            bins, dup_last(y), label=label, facecolor=color, step="post", alpha=0.2
        )
    else:
        ax.step(
            bins,
            dup_last(y),
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
            ls=linestyle,
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
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        ax.step(
            bins,
            dup_last(y_low),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        ax.fill_between(
            bins,
            dup_last(y_low),
            dup_last(y_high),
            facecolor=color,
            alpha=0.3,
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