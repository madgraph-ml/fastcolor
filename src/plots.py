from utils.plots_utils import *

TRUTH_COLOR = "#07078A"
NEUTRAL_COLOR = "black"
NN_COLOR_red = "#8A0707"
NN_COLOR_green = "#06793F"
NN_COLOR_purple = "#790679"
NN_COLORS = {
    "MLP": NN_COLOR_red,
    "Transformer": NN_COLOR_green,
    "LGATr": NN_COLOR_purple,
}

class Plots:
    """
    Implements the plotting pipeline to evaluate the performance of
    conditional generative networks.
    Args:
        logger: Logger object for logging messages
        dataset: Dataset object containing the data to be plotted
        losses: Dictionary of training losses
        dataset_loss: Dictionary of [trn, tst, val] losses for the dataset
        process_name: Name of the process being analyzed, optional
        regress: Type of regression used, e.g., "r", "LC", "FC"
        debug: If True, enables debug mode with additional logging
        model_name: Name of the model used for predictions, optional
        loss_name: Name of the loss function used, optional
    """

    def __init__(
        self,
        logger,
        dataset,
        losses: dict = None,
        dataset_loss: dict = None,
        process_name: str = None,
        regress: str = None,
        debug: bool = False,
        model_name: str = None,
        loss_name: str = None,
    ):
        self.logger = logger
        self.dataset = dataset
        self.channels = dataset.channels
        self.observables = dataset.observables
        self.bins = dataset.bins
        self.obs = dataset.obs
        self.obs_ppd = dataset.obs_ppd if hasattr(dataset, "obs_ppd") else None
        if process_name is not None:
            if process_name in ["gg_4g", "gg_5g", "gg_6g", "gg_7g"]:
                process_name = f"${process_name[:2]}\\to {process_name[-2:]}$"
            elif process_name in [
                "gg_qqbar2g",
                "gg_qqbar3g",
                "gg_qqbar4g",
                "gg_qqbar5g",
            ]:
                process_name = (
                    f"${process_name[:2]} \\to q\\bar{{q}} + {process_name[-2:]}$"
                )
            elif process_name == "gg2aag":
                process_name = f"$gg \\to \\gamma \\gamma + g$"
            elif process_name == "new":
                process_name = f"$gg \\to 4g$ (new)"
            elif process_name == "rik_amps":
                process_name = f"$gg \\to 4g$"
            elif process_name == "gg_4g_ext_comb":
                process_name = f"$gg \\to 4g$ "
            elif process_name == "gg_5g_ext_comb":
                process_name = f"$gg \\to 5g$ "
        if loss_name is not None:
            process_name = f"{process_name} ({ {'heteroschedastic': 'het', 'MSE': 'MSE'}[loss_name] })" if process_name is not None else loss_name
        self.losses = losses
        self.metrics = dataset_loss if dataset_loss is not None else None

        self.process_name = process_name
        self.regress_name = {
            "r" : "r",
            "LC" : "A_{\\text{LC}}",
            "FC" : "A_{\\text{FC}}",
        }[regress]
        self.regress = regress
        self.debug = debug
        self.model_name = model_name if model_name is not None else "NN"

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
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.15, 0.1, 0.85, 0.98))
            for loss, label in zip([self.losses["trn"], self.losses["val"]], labels):
                if len(loss) == len(iterations):
                    its = iterations
                else:
                    frac = len(self.losses["trn"]) / len(loss)
                    its = np.arange(1, len(loss) + 1) * frac
                ax.plot(its, loss, label=label)
            if logy:
                ax.set_yscale("log") if np.all(np.array(self.losses["trn"]) > 0) else ax.set_yscale("symlog")
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
                    pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.15, 0.1, 0.85, 0.98)
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
                ax.set_yscale("log")
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Gradient norm")
                ax.legend(frameon=False, loc="upper right")
                fig.savefig(pp, format="pdf")
                plt.close()
            if "hs" in self.losses:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                fig.tight_layout(
                    pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.15, 0.1, 0.85, 0.98)
                )
                iterations = range(1, len(self.losses["hs"]) + 1)
                ax.plot(
                    iterations,
                    self.losses["hs"],
                    color="darkorange",
                    label=f"${self.losses['hs_scale']}\cdot\sigma(\hat{{{self.regress_name}}}/{{{self.regress_name}}})$",
                )
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.set_yscale("log")
                ax.set_xlabel("Number of iterations")
                ax.set_ylabel("Variance loss")
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
                self.observables,
                self.bins,
                self.obs,
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
                hist_plot(
                    pp,
                    lines,
                    bins,
                    obs,
                    show_ratios=False,
                    title=self.process_name if self.process_name is not None else None,
                    model_name=self.model_name,
                )
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
        with PdfPages(file) as pp:
            for obs, data in zip(
                self.observables,
                self.obs_ppd,
            ):

                xlim_bins = np.percentile(data, [0.5, 99.5])
                xlim_bins[0] = 1.1 * xlim_bins[0] if xlim_bins[0] < 0 else 0.91 * xlim_bins[0]
                xlim_bins[1] = 0.91 * xlim_bins[1] if xlim_bins[1] < 0 else 1.1 * xlim_bins[1]
                bins = np.linspace(*np.array(xlim_bins), 51)
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
                hist_plot(
                    pp,
                    lines,
                    bins,
                    obs,
                    show_ratios=False,
                    title=self.process_name if self.process_name is not None else None,
                    model_name=self.model_name,
                )
                if pickle_file is not None:
                    pickle_data.append({"lines": lines, "bins": bins, "obs": obs})
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_data, f)

    def plot_weights(self, file: str, split="tst", ppd: bool = False, percentage_of_ratio_data: float = 100., pickle_file: Optional[str] = None, fix_bins: bool = False):
        """
        Makes plots of the weights learned for Pythia vs Herwig.
        Args:
            file: Output file name
        """
        if not ppd:
            reweight_factors_truth = (
                self.dataset.events[split][:, -1].detach().cpu().numpy()
            )
            reweight_factors_pred = (
                self.dataset.predicted_factors_raw[split].squeeze().detach().cpu().numpy()
            )
        else:
            reweight_factors_truth = (
                    self.dataset.events_ppd[split][:, -1].squeeze().detach().cpu().numpy()
                )
            reweight_factors_pred = (
                self.dataset.predicted_factors_ppd[split].squeeze().detach().cpu().numpy()
            )
        
        metrics = self.metrics[split] if not ppd else None
        if fix_bins:
            self.logger.info(f"         Fixing bins for weights plots, ppd={ppd}")
            percentage_of_ratio_data = -1
            if self.regress == "r":
                bins_targets = np.linspace(
                        min(reweight_factors_truth) - 0.01 * np.abs(min(reweight_factors_truth)), max(reweight_factors_truth) + 0.01 * np.abs(max(reweight_factors_truth)), 64
                    )
                if not ppd:
                    bins_ratios = np.linspace(
                        0.95, 1.05, 64
                    )
                    bins_deltas = np.linspace(
                        -0.07, 0.07, 64
                    )
                    bins_abs_deltas = np.logspace(
                        -9, -1, 64
                    )
                else:
                    bins_ratios = np.linspace(
                        -20, 20, 64
                    )
                    bins_deltas = np.linspace(
                        -20, 20, 64
                    )
                    bins_abs_deltas = np.logspace(
                        -8, 5, 64
                    )
            else:
                if not ppd:
                    bins_targets = np.logspace(
                        np.log10(min(reweight_factors_truth) - 0.1 * np.abs(min(reweight_factors_truth))),
                        np.log10(max(reweight_factors_truth) + 0.1 * np.abs(max(reweight_factors_truth))),
                        64,
                    )
                    bins_ratios = np.logspace(
                        -2, 2, 64
                    )
                    bins_deltas = np.linspace(
                        -3, 20, 64
                    )
                    bins_abs_deltas = np.logspace(
                        -9, 5, 64
                    )
                else:
                    bins_targets = np.linspace(
                        min(reweight_factors_truth) - 0.1 * np.abs(min(reweight_factors_truth)), max(reweight_factors_truth) + 0.1 * np.abs(max(reweight_factors_truth)), 64
                    )
                    bins_ratios = np.linspace(
                        -50, 50, 64
                    )
                    bins_deltas = np.linspace(
                        -50, 50, 64
                    )
                    bins_abs_deltas = np.logspace(
                        -8, 5, 64
                    )
        else:
            bins_targets = None
            bins_ratios = None
            bins_deltas = None
            bins_abs_deltas = None

        with PdfPages(file) as pp:
            self._plot_targets(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                pickle_file=pickle_file,
                metrics = metrics,
                bins = bins_targets
            )
            self._plot_ratios(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                metrics=metrics,
                bins = bins_ratios
            )
            self._plot_deltas(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                metrics=metrics,
                bins = bins_deltas
            )
            self._plot_deltas(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                abs=True,
                metrics=metrics,
                bins = bins_abs_deltas
            )

    def _plot_targets(self, pp: PdfPages, pred: np.ndarray, truth: np.ndarray, ppd: bool = False, pickle_file: str = None, metrics: np.ndarray = None, bins: Optional[np.ndarray] = None) -> None:
        """
        Makes a plot of the regressed factors, whichever they are.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """
        xlim_bins = np.array([0.9*min(min(truth), min(pred)), 1.1*max(max(truth), max(pred))])
        xlim_bins[0] = 1.1 * xlim_bins[0] if xlim_bins[0] < 0 else xlim_bins[0]
        if bins is None:
            bins = (
            np.linspace(*xlim_bins, 64)
            if self.regress == "r" or ppd else
            np.logspace(*np.log10(xlim_bins), 64)
            if np.all(xlim_bins > 0) else
            np.logspace(np.log10(max(1e-10, xlim_bins[0])), np.log10(max(1e-11, xlim_bins[1])), 64)
        )
        y_truth, y_truth_err = compute_hist_data(
            bins, truth, bayesian=False
        )
        y_pred, y_pred_err = compute_hist_data(
            bins, pred, bayesian=False
        )

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
                label=f"{self.model_name}",
                color=NN_COLORS[self.model_name],
            ),
        ]
        hist_weights_plot(
            pp,
            lines,
            bins,
            show_ratios=True,
            title=self.process_name if self.process_name is not None else None,
            xlabel=f"${{{self.regress_name}}}(x)$" if not ppd else f"$\\tilde{{{self.regress_name}}}(x)$",
            xscale="log" if not self.regress == "r" and not ppd else "linear",
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {
                "targets-lines": lines,
                "targets-bins": bins
            }
            append_to_pickle(pickle_file, pickle_data)

    def _plot_ratios(self, pp: PdfPages, pred: np.ndarray, truth: np.ndarray, ppd: bool = False, percentage_of_ratio_data: float = 100., pickle_file: str = None, eps = 1e-20, metrics: np.ndarray = None, bins: Optional[np.ndarray] = None) -> None:
        """
        Makes a plot of the ratio of the truth and predicted distributions.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """
        adjusted_pred = np.where(pred > 0, pred + eps, pred - eps)
        ratios = truth / adjusted_pred
        if bins is None:
            xlim_bins = np.percentile(ratios, [50 - percentage_of_ratio_data/2, 50 + percentage_of_ratio_data/2])
            bins = (
            np.linspace(*xlim_bins, 64)
            if self.regress == "r" or ppd else
            np.logspace(*np.log10(xlim_bins), 64)
            if np.all(xlim_bins > 0) else
            np.logspace(np.log10(max(1e-11, xlim_bins[0])), np.log10(max(1e-11, xlim_bins[1])), 64)
        )


        y_diff, y_diff_err = compute_hist_data(bins, ratios, bayesian=False)
        lines = [
            Line(
                y=y_diff,
                y_err=y_diff_err,
                label=f"$\\text{{{self.model_name}}}$",
                color=NN_COLORS[self.model_name],
            ),
        ]

        if percentage_of_ratio_data >= 0:
            perc_str = f"({percentage_of_ratio_data:.0f}\\%)"
        else:
            perc_str = ""

        if ppd:
            label = fr"$\tilde{{{self.regress_name}}}^{{\text{{truth}}}} / \tilde{{{self.regress_name}}}^{{\text{{pred}}}}{perc_str}$"
        else:
            label = fr"${self.regress_name}^{{\text{{truth}}}} / {self.regress_name}^{{\text{{pred}}}}{perc_str}$"

        hist_weights_plot(
            pp,
            lines,
            bins,
            show_ratios=False,
            xlabel=label,
            xscale="linear" if self.regress == "r" or ppd else "log",
            title=self.process_name if self.process_name is not None else None,
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {
                "ratios-lines": lines,
                "ratios-bins": bins
            }
            append_to_pickle(pickle_file, pickle_data)
    
    def _plot_deltas(self, pp: PdfPages, pred: np.ndarray, truth: np.ndarray, ppd: bool = False, percentage_of_ratio_data: float = 100., pickle_file: str = None, eps = 1e-20, abs = False, metrics: np.ndarray = None, bins: Optional[np.ndarray] = None) -> None:
        """
        Makes a plot of the delta of the truth and predicted distributions.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """

        adjusted_truth = np.where(truth > 0, truth + eps, truth - eps)
        delta = (pred - truth) / adjusted_truth
        delta = np.abs(delta) if abs else delta  # take absolute value if abs is True
        # xlim_bins = np.array([0.9*min(delta), 1.1*max(delta)])
        if bins is None:
            xlim_bins = np.percentile(delta, [50 - percentage_of_ratio_data/2, 50 + percentage_of_ratio_data/2])
            if not abs:
                bins = np.linspace(*xlim_bins, 64)
            else:
                bins = np.logspace(*np.log10(np.array(xlim_bins)), 64)

        y_delta, y_delta_err = compute_hist_data(
            bins, delta, bayesian=False
        )
        lines = [
            Line(
                y=y_delta,
                y_err=y_delta_err,
                label=f"{self.model_name}",
                color=NN_COLORS[self.model_name],
            ),
        ]
                # Build percentage string if needed
        perc_str = f"({percentage_of_ratio_data:.0f}\\%)" if percentage_of_ratio_data >= 0 else ""

        # Build delta expression
        if ppd:
            delta_expr = fr"\tilde{{\Delta}}_{{{self.regress_name}}} = \frac{{\tilde{{{self.regress_name}}}^{{\text{{pred}}}} - \tilde{{{self.regress_name}}}^{{\text{{true}}}}}}{{\tilde{{{self.regress_name}}}^{{\text{{true}}}}}}"
        else:
            delta_expr = fr"\Delta_{{{self.regress_name}}} = \frac{{{self.regress_name}^{{\text{{pred}}}} - {self.regress_name}^{{\text{{true}}}}}}{{{self.regress_name}^{{\text{{true}}}}}}"

        # Final xlabel
        xlabel = f"${delta_expr}{perc_str}$"

        hist_weights_plot(
            pp,
            lines,
            bins,
            show_ratios=False,
            xlabel=xlabel if not abs else f"$|{{\Delta}}_{{{self.regress_name}}}|$" if not ppd else f"$|\\tilde{{\Delta}}_{{{self.regress_name}}}|$",
            xscale="log" if abs else "linear",
            title=self.process_name if self.process_name is not None else None,
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {
                f"deltas{'_abs' if abs else ''}-lines": lines,
                f"deltas{'_abs' if abs else ''}-bins": bins
            }
            append_to_pickle(pickle_file, pickle_data)

    def plot_ratio_correlation(
        self, file: str, split="tst", ppd: bool = False, percentage_of_ratio_data: float = 100., pickle_file: Optional[str] = None, eps = 1e-20
    ):
        with PdfPages(file) as pp:
            cmap = plt.get_cmap("viridis")
            cmap.set_bad("white")
            if not ppd:
                reweight_factors_truth = (
                    self.dataset.events[split][:, -1].squeeze().detach().cpu().numpy()
                )
                reweight_factors_pred = (
                    self.dataset.predicted_factors_raw[split].squeeze().detach().cpu().numpy()
                )
            else:
                reweight_factors_truth = (
                    self.dataset.events_ppd[split][:, -1].squeeze().detach().cpu().numpy()
                )
                reweight_factors_pred = (
                    self.dataset.predicted_factors_ppd[split].squeeze().detach().cpu().numpy()
                )
            adjusted_factors_pred = np.where(reweight_factors_pred > 0, reweight_factors_pred + eps, reweight_factors_pred - eps)
            ratios = reweight_factors_truth / adjusted_factors_pred
            xlim_bins = [
                np.array([0.9*min(ratios), 1.1*max(ratios)]),
                np.array([0.9 * min(reweight_factors_pred), 1.1 * max(reweight_factors_pred)]),
            ]

            xlim_bins[0] = np.percentile(ratios, [50 - percentage_of_ratio_data/2, 50 + percentage_of_ratio_data/2]) # to contain percentage_of_ratio_data% of the data (approx bc later is modified slightly)
            # correct limits if min or max are negative
            for i in (0, 1):
                lo, hi = xlim_bins[i]
                if lo < 0:
                    lo *= 1.1
                if hi < 0:
                    hi *= 0.91
                xlim_bins[i] = np.array([lo, hi])
            bins1 = [
                np.linspace(*xlim_bins[0], 64) if self.regress == "r" or ppd else np.logspace(*np.log10(xlim_bins[0]), 64) if np.all(xlim_bins[0] > 0) else np.logspace(np.log10(max(1e-11, xlim_bins[0][0])), np.log10(max(1e-11, xlim_bins[0][1])), 64),
                np.linspace(*xlim_bins[1], 64) if self.regress == "r" or ppd else np.logspace(*np.log10(xlim_bins[1]), 64) if np.all(xlim_bins[1] > 0) else np.logspace(np.log10(max(1e-11, xlim_bins[1][0])), np.log10(max(1e-11, xlim_bins[1][1])), 64),
            ]
            h, x, y = np.histogram2d(ratios, reweight_factors_pred, bins=(bins1[0], bins1[1]))
            # h = np.ma.divide(h, np.sum(h, -1, keepdims=True)).filled(0)
            h[h == 0] = np.nan
            h_norm1 = h / np.nansum(h)
            fig, ax = plt.subplots(figsize=(5, 5))
            norm1 = safe_lognorm(h_norm1.T)
            pcm = ax.pcolormesh(
                bins1[0],
                bins1[1],
                h_norm1.T,
                cmap=cmap,
                norm=norm1,
                rasterized=True,
            )


            fig.colorbar(pcm, ax=ax)
            fig.suptitle(f"{self.model_name}")
            ax.set_xscale("linear" if self.regress == "r" or ppd else "log")
            ax.set_yscale("linear" if self.regress == "r" or ppd else "log")
            ax.set_xlim(bins1[0][0], bins1[0][-1])
            ax.set_ylim(bins1[1][0], bins1[1][-1])
            ax.set_xlabel(
                f"${self.regress_name}^{{\\text{{truth}}}}/{self.regress_name}^{{\\text{{pred}}}}({percentage_of_ratio_data:.0f}\\%)$"
                if not ppd else
                f"$\\tilde{{{self.regress_name}}}^{{\\text{{truth}}}}/\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}({percentage_of_ratio_data:.0f}\\%)$"
            )
            ax.set_ylabel(f"${self.regress_name}^{{\\text{{pred}}}}$" if not ppd else f"$\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}$")
            fig.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


            xlim_bins = [
                np.array([0.9*min(reweight_factors_truth), 1.1*max(reweight_factors_truth)]),
                np.array([0.9 * min(adjusted_factors_pred), 1.1 * max(adjusted_factors_pred)]),
            ]

            xlim_bins[0] = np.percentile(reweight_factors_truth, [50 - percentage_of_ratio_data/2, 50 + percentage_of_ratio_data/2]) # to contain percentage_of_ratio_data% of the data (approx bc later is modified slightly)
            # correct limits if min or max are negative
            for i in (0, 1):
                lo, hi = xlim_bins[i]
                if lo < 0:
                    lo *= 1.1
                if hi < 0:
                    hi *= 0.91
                xlim_bins[i] = np.array([lo, hi])
            bins2 = [
                np.linspace(*xlim_bins[0], 64) if self.regress == "r" or ppd else np.logspace(*np.log10(xlim_bins[0]), 64) if np.all(xlim_bins[0] > 0) else np.logspace(np.log10(max(1e-11, xlim_bins[0][0])), np.log10(max(1e-11, xlim_bins[0][1])), 64),
                np.linspace(*xlim_bins[1], 64) if self.regress == "r" or ppd else np.logspace(*np.log10(xlim_bins[1]), 64) if np.all(xlim_bins[1] > 0) else np.logspace(np.log10(max(1e-11, xlim_bins[1][0])), np.log10(max(1e-11, xlim_bins[1][1])), 64),
            ]
            h, x, y = np.histogram2d(reweight_factors_truth, reweight_factors_pred, bins=(bins2[0], bins2[1]))
            # h = np.ma.divide(h, np.sum(h, -1, keepdims=True)).filled(0)
            h[h == 0] = np.nan
            h_norm2 = h / np.nansum(h)
            fig, ax = plt.subplots(figsize=(5, 5))
            norm2 = safe_lognorm(h_norm2.T)
            pcm = ax.pcolormesh(
                bins2[0],
                bins2[1],
                h_norm2.T,
                cmap=cmap,
                norm=norm2,
                rasterized=True,
            )
            ax.plot(
                [max(bins2[0][0], bins2[1][0]), min(bins2[0][-1], bins2[1][-1])],  # limits of x=y within current axes
                [max(bins2[0][0], bins2[1][0]), min(bins2[0][-1], bins2[1][-1])],
                linestyle='--',
                color='gray',
                linewidth=2,
            )


            fig.colorbar(pcm, ax=ax)
            fig.suptitle(f"{self.model_name}")
            ax.set_xscale("linear" if self.regress == "r" or ppd else "log")
            ax.set_yscale("linear" if self.regress == "r" or ppd else "log")
            ax.set_xlim(bins2[0][0], bins2[0][-1])
            ax.set_ylim(bins2[1][0], bins2[1][-1])
            ax.set_xlabel(
                f"${self.regress_name}^{{\\text{{truth}}}}({percentage_of_ratio_data:.0f}\\%)$"
                if not ppd else
                f"$\\tilde{{{self.regress_name}}}^{{\\text{{truth}}}}({percentage_of_ratio_data:.0f}\\%)$"
            )
            ax.set_ylabel(f"${self.regress_name}^{{\\text{{pred}}}}$" if not ppd else f"$\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}$")
            fig.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()

            
            if pickle_file is not None:
                pickle_data = {
                    "ratio_vs_pred-h_norm.T": h_norm1.T,
                    "ratio_vs_pred-bins": bins1,
                    "ratio_vs_pred-norm" : norm1,
                    "pred_vs_truth-h_norm.T": h_norm2.T,
                    "pred_vs_truth-bins": bins2,
                    "pred_vs_truth-norm": norm2
                }
                append_to_pickle(pickle_file, pickle_data)