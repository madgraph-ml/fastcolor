import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
from madrecolor.datasets.dataset import Observable
from typing import Optional

TRUTH_COLOR = "#07078A"
NN_COLOR = "#8A0707"
NEUTRAL_COLOR="black"

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


class Plots:
    """
    Implements the plotting pipeline to evaluate the performance of
    conditional generative networks.
    Args:
        logger: Logger object
        observables: List of observables
        losses: Dictionary with loss terms and learning rate as a function of the epoch
        x_part: True data at particle level
        x_reco: True data at reconstruction level
        x_gen: Generated particle level data
        x_part_ppd: Preprocessed particle level data
        x_reco_ppd: Preprocessed reconstruction level data
        debug: If True, additional debug information is printed
    """

    def __init__(
        self,
        logger,
        dataset,
        losses: dict = None,
        debug: bool = False,
    ):
        self.logger = logger
        self.dataset = dataset
        self.channels = dataset.channels
        self.observables = dataset.observables
        self.bins = dataset.bins
        self.obs = dataset.obs
        self.obs_ppd = (
            dataset.obs_ppd if hasattr(dataset, "obs_ppd") else None
        )
        self.losses = losses
        self.debug = debug

    plt.rc("font", family="serif", size=16)
    plt.rc("font", serif="Charter")
    plt.rc("axes", titlesize="medium")
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
    plt.rc("text", usetex=True)

    def plot_train_metrics(self, file: str, labels=None, logy=False, ylabel="Loss"):
        with PdfPages(file) as pp:
            iterations = range(1, len(self.losses["trn"]) + 1)
            lr = self.losses.get("lr", None)
            labels = ["Train", "Validation"] if labels is None else labels
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.tight_layout(
                pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.15, 0.1, 0.85, 0.98)
            )
            for loss, label in zip([self.losses["trn"], self.losses["val"]], labels):
                if len(loss) == len(iterations):
                    its = iterations
                else:
                    frac = len(self.losses["trn"]) / len(loss)
                    its = np.arange(1, len(loss) + 1) * frac
                ax.plot(its, loss, label=label)
            if logy:
                ax.set_yscale("log")
            if lr is not None:
                axright = ax.twinx()
                axright.plot(iterations, lr, label="Learning Rate", color="crimson")
                axright.set_ylabel("Learning rate")
                axright.spines["right"].set_color("crimson")
                axright.yaxis.label.set_color("crimson")
                axright.tick_params(axis="y", colors="crimson")
            ax.set_xlabel("Number of iterations")
            ax.set_ylabel(ylabel)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.legend(
                ax.get_legend_handles_labels()[0]
                + axright.get_legend_handles_labels()[0],
                ax.get_legend_handles_labels()[1]
                + axright.get_legend_handles_labels()[1],
                frameon=False,
                loc="upper right",
            )
            fig.savefig(pp, format="pdf")
            plt.close()

            if "grad_norm" in self.losses:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                fig.tight_layout(
                    pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.09, 0.1, 0.85, 0.98)
                )
                iterations = range(1, len(self.losses["grad_norm"]) + 1)
                ax.plot(
                    iterations,
                    self.losses["grad_norm"],
                    color="firebrick",
                    label="Gradient norm",
                )
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Gradient norm")
                ax.legend(frameon=False, loc="upper right")
                fig.savefig(pp, format="pdf")
                plt.close()

    def plot_observables(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        pickle_data = []
        with PdfPages(file) as pp:
            for obs, bins, data in zip(
                self.observables, self.bins, self.obs,
            ):
                y, y_err = compute_hist_data(bins, data, bayesian=False)

                lines = [
                    Line(
                        y=y,
                        y_err=y_err,
                        label="Events",
                        color=TRUTH_COLOR,
                    ),
                ]
                hist_plot(pp, lines, bins, obs, show_ratios=True)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)


    def plot_observables_ppd(self, file: str, pickle_file: Optional[str] = None):
        """
        Makes histograms of truth and generated distributions for all observables.
        Args:
            file: Path to the output PDF file
            pickle_file: Path to the output pickle file (optional)
        """
        pickle_data = []
        bins_ppd = [
            np.linspace(-5, 5, 51) for j in self.bins
        ]  # change plot bins for preprocessed features
        with PdfPages(file) as pp:
            for obs, bins, data in zip(
                self.observables,
                bins_ppd,
                self.obs_ppd,
            ):
                # if obs.channel not in self.channels:
                #     # avoid plotting primary channels not used for unfolding
                #     continue
                y, y_err = compute_hist_data(bins, data, bayesian=False)

                lines = [
                    Line(
                        y=y,
                        y_err=y_err,
                        label="Events",
                        color=TRUTH_COLOR,
                    ),
                ]
                hist_plot(pp, lines, bins, obs, show_ratios=True)
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)


    def plot_weights(self, file: str, split='tst', pickle_file: Optional[str] = None):
            """
            Makes plots of the weights learned for Pythia vs Herwig.
            Args:
                file: Output file name
            """
            with PdfPages(file) as pp:
                reweight_factors_truth = self.dataset.events[split][:, -1]
                reweight_factors_pred = self.dataset.predicted_factors_raw[split]
                
                xlim_bins = [0.8, 1.4]
                bins = np.linspace(*xlim_bins, 64)
                y_truth, y_truth_err = compute_hist_data(bins, reweight_factors_truth, bayesian=False)
                y_pred, y_pred_err = compute_hist_data(bins, reweight_factors_pred, bayesian=False)

                lines = [
                            Line(
                                y=y_truth,
                                y_err=y_truth_err,
                                label="Truth",
                                color=TRUTH_COLOR,
                            ),
                            Line(
                                y=y_pred,
                                y_err=y_pred_err,
                                y_ref=y_truth,
                                label="NN",
                                color=NN_COLOR,
                            ),
                        ]
                self.hist_weights_plot(pp, lines, bins, show_ratios=True)

                # differences = reweight_factors_pred - reweight_factors_truth
                # # xlim_bins = [-., .5]
                # bins = np.linspace(*xlim_bins, 64)
                # y_diff, y_diff_err = compute_hist_data(bins, differences, bayesian=False)
                # lines = [
                #             Line(
                #                 y=y_diff,
                #                 y_err=y_diff_err,
                #                 label="Difference",
                #                 color=NEUTRAL_COLOR,
                #             ),
                #         ]
                # self.hist_weights_plot(pp, lines, bins, show_ratios=False)

    def plot_weights_ppd(self, file: str, split='tst', pickle_file: Optional[str] = None):
        """
        Makes plots of the weights learned for Pythia vs Herwig.
        Args:
            file: Output file name
        """
        with PdfPages(file) as pp:
            reweight_factors_truth = self.dataset.events_ppd[split][:, -1]
            reweight_factors_pred = self.dataset.predicted_factors_ppd[split]
            
            xlim_bins = [-5, 5]
            bins = np.linspace(*xlim_bins, 64)
            y_truth, y_truth_err = compute_hist_data(bins, reweight_factors_truth, bayesian=False)
            y_pred, y_pred_err = compute_hist_data(bins, reweight_factors_pred, bayesian=False)

            lines = [
                        Line(
                            y=y_truth,
                            y_err=y_truth_err,
                            label="Truth",
                            color=TRUTH_COLOR,
                        ),
                        Line(
                            y=y_pred,
                            y_err=y_pred_err,
                            y_ref=y_truth,
                            label="NN",
                            color=NN_COLOR,
                        ),
                    ]
            self.hist_weights_plot(pp, lines, bins, show_ratios=True)

            # differences = reweight_factors_pred - reweight_factors_truth
            # # xlim_bins = [-., .5]
            # bins = np.linspace(*xlim_bins, 64)
            # y_diff, y_diff_err = compute_hist_data(bins, differences, bayesian=False)
            # lines = [
            #             Line(
            #                 y=y_diff,
            #                 y_err=y_diff_err,
            #                 label="Difference",
            #                 color=NEUTRAL_COLOR,
            #             ),
            #         ]
            # self.hist_weights_plot(pp, lines, bins, show_ratios=False)


    
    def hist_weights_plot(
        self,
        pdf: PdfPages,
        lines: list[Line],
        bins: np.ndarray,
        show_ratios: bool = False,
        title: Optional[str] = None,
        no_scale: bool = False,
        yscale: Optional[str] = None,
        show_metrics: bool = False,
        ylim: tuple[float, float] = None,
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

            n_panels = 1 + int(show_ratios) + int(show_metrics)
            fig, axs = plt.subplots(
                n_panels,
                1,
                sharex=True,
                figsize=(6, 4.5),
                gridspec_kw={"height_ratios": (12, 3, 1)[:n_panels], "hspace": 0.00},
            )
            if n_panels == 1:
                axs = [axs]

            for line in lines:
                if line.vline:
                    axs[0].axvline(line.y, label=line.label, color=line.color, linestyle=line.linestyle)
                    continue
                integral = np.sum((bins[1:] - bins[:-1]) * line.y)
                scale = 1 / integral if integral != 0.0 else 1.0
                if line.y_ref is not None:
                    ref_integral = np.sum((bins[1:] - bins[:-1]) * line.y_ref)
                    ref_scale = 1 / ref_integral if ref_integral != 0.0 else 1.0
                if no_scale:
                    scale = 1.
                    ref_scale = 1.

                hist_line(
                    axs[0],
                    bins,
                    line.y * scale,
                    line.y_err * scale if line.y_err is not None else None,
                    label=line.label,
                    color=line.color,
                    fill=line.fill,
                    linestyle=line.linestyle
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
                axs[1].set_ylabel(r"$\frac{\mathrm{NN}}{\mathrm{Truth}}$")
                axs[1].set_yticks([0.9, 1, 1.1])
                axs[1].set_ylim([0.85, 1.15])
                axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
                axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
                axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

            axs[0].legend(frameon=False)
            axs[0].set_ylabel("Normalized")
            axs[0].set_yscale("log" if yscale is None else yscale)
            if ylim is not None:
                axs[0].set_ylim(*ylim)
            if title is not None:
                self.corner_text(axs[0], title, "left", "top")

            axs[-1].set_xlabel(f"$w(x)$")
            # axs[-1].set_xscale("log")
            axs[-1].set_xlim(bins[0], bins[-1])
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            plt.close()



def compute_hist_data(bins: np.ndarray, data: np.ndarray, bayesian=False, weights=None):
    if bayesian:
        hists = np.stack(
            [
                np.histogram(d, bins=bins, density=False, weights=weights)[0]
                for d in data
            ],
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

        axs[0].legend(
            frameon=False, **(legend_kwargs if legend_kwargs is not None else {})
        )
        axs[0].set_ylabel("Normalized")
        axs[0].set_yscale(observable.yscale if yscale is None else yscale)
        if title is not None:
            corner_text(axs[0], title, "left", "top")

        if show_ratios:
            axs[1].set_ylabel(r"$\frac{\mathrm{Model}}{\mathrm{Truth}}$")
            axs[1].set_yticks([0.9, 1, 1.1])
            axs[1].set_ylim([0.85, 1.15])
            axs[1].axhline(y=1, c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.1, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.9, c="black", ls="dotted", lw=0.5)

        unit = "" if observable.unit is None else f" [{observable.unit}]"
        axs[-1].set_xlabel(f"${{{observable.tex_label}}}${unit}")
        axs[-1].set_xscale(observable.xscale)
        axs[-1].set_xlim(bins[0], bins[-1])
        plt.savefig(pdf, format="pdf")
        plt.close()


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





    # def plot_multiplicity_cut_observables(self, file: str, pickle_file: Optional[str] = None):
    #     """
    #     Makes histograms of truth and generated distributions for observables cut on jet/quark multiplicity.
    #     Args:
    #         file: Path to the output PDF file
    #         pickle_file: Path to the output pickle file (optional)
    #     """
    #     pickle_data = []
    #     with PdfPages(file) as pp:
    #         for obs, bins, data_hard, data_reco, data_unf in zip(
    #             self.observables, self.bins, self.obs_part, self.obs_reco, self.obs_unf
    #         ):

    #             if "p_{T, j" in obs.tex_label or "p_{T, q" in obs.tex_label:
    #                 for i in range(2, 9):
    #                     hard_mults = data_hard[self.obs_part[20] == i] if self.unfold_to=="particle" else data_hard[self.obs_reco[20] == i]
    #                     reco_mults = data_reco[self.obs_reco[20] == i]
    #                     unf_mults = data_unf[self.obs_reco[20] == i]
                
    #                     y_hard, y_hard_err = compute_hist_data(bins, hard_mults, bayesian=False)
    #                     y_reco, y_reco_err = compute_hist_data(bins, reco_mults, bayesian=False)
    #                     y_unf, y_unf_err = compute_hist_data(bins, unf_mults, bayesian=False)

    #                     lines = [
    #                         Line(
    #                             y=y_reco,
    #                             y_err=y_reco_err,
    #                             label="Reco",
    #                             color=RECO_COLOR,
    #                         ),
    #                         Line(
    #                             y=y_hard,
    #                             y_err=y_hard_err,
    #                             label="Part" if self.unfold_to=="particle" else "Hard",
    #                             color=PART_COLOR,
    #                         ),
    #                         Line(
    #                             y=y_unf,
    #                             y_err=y_unf_err,
    #                             y_ref=y_hard,
    #                             label="CFM",
    #                             color=CFM_COLOR,
    #                         ),
    #                     ]
    #                     hist_plot(pp, lines, bins, obs, show_ratios=True, title=f"${i}j$")
    #                     if pickle_file is not None:
    #                         pickle_data.append({"lines": lines, "bins": bins, "obs": obs})
    #     if pickle_file is not None:
    #         with open(pickle_file, "wb") as f:
    #             pickle.dump(pickle_data, f)


    # def plot_migration(
    #     self, file: str, gt_part=False, pickle_file: Optional[str] = None
    # ):
    #     if gt_part:
    #         obs_part = self.obs_part
    #         name_part = "Part" if self.unfold_to=="particle" else "Hard"
    #     else:
    #         obs_part = self.obs_unf
    #         name_part = "Unfold"
    #     ranges = [[0, 0.4], [0, 0.3], [0, 0.5], [0, 0.5], [0, 0.2], [0, 0.25]]
    #     ranges += [[0, 0.5] for _ in range(len(self.observables) - len(ranges))]

    #     pickle_data = []
    #     with PdfPages(file) as pp:
    #         for k, (obs, bins, data_hard, data_reco, r) in enumerate(
    #             zip(self.observables, self.bins, obs_part, self.obs_reco, ranges)
    #         ):
    #             if obs.channel is not None and obs.channel not in self.channels:
    #                 # avoid plotting primary (obs.channel != None) channels not used for unfolding
    #                 # secondary channels (derived observables -- obs.channel == None) are always plotted
    #                 continue
    #             cmap = plt.get_cmap("viridis")
    #             cmap.set_bad("white")
    #             h, x, y = np.histogram2d(data_hard, data_reco, bins=(bins, bins))
    #             h = np.ma.divide(h, np.sum(h, 0, keepdims=True)).filled(
    #                 0
    #             )  # normalize so each column sums up to 1
    #             h[h == 0] = np.nan
    #             plt.pcolormesh(
    #                 bins, bins, h, cmap=cmap, rasterized=True, vmin=r[0], vmax=r[1]
    #             )
    #             plt.colorbar()

    #             unit = "" if obs.unit is None else f" [{obs.unit}]"
    #             plt.title(f"${{{obs.tex_label}}}${unit}")
    #             plt.xlim(bins[0], bins[-1])
    #             plt.ylim(bins[0], bins[-1])
    #             plt.xlabel("Reco")
    #             plt.ylabel(name_part)
    #             plt.savefig(pp, format="pdf", bbox_inches="tight")
    #             plt.close()

    #             if pickle_file is not None:
    #                 pickle_data.append({"h": h, "bins": bins, "ranges": r, "obs": obs})
    #     if pickle_file is not None:
    #         with open(pickle_file, "wb") as f:
    #             pickle.dump(pickle_data, f)