from src.utils.plots_utils import *
from src.utils.paths_dict import paths as paths_dict

TRUTH_COLOR = "#07078A"
NEUTRAL_COLOR = "black"
NN_COLOR_red = "#8A0707"
NN_COLOR_green = "#06793F"
NN_COLOR_purple = "#790679"
NN_COLOR_orange = "darkorange"
NN_COLORS = {
    "MLP": NN_COLOR_red,
    "Transformer": NN_COLOR_green,
    "L-GATr": NN_COLOR_purple,
    "GNN": NN_COLOR_orange,
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
        metrics: dict = None,
        process: str = None,
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
        if process is not None:
            if process in ["gg_4g", "gg_5g", "gg_6g", "gg_7g"]:
                process_name = f"${process[:2]}\\to {process[-2:]}$"
            elif process in [
                "gg_ddbar2g",
                "gg_ddbar3g",
                "gg_ddbar4g",
                "gg_ddbar5g",
            ]:
                process_name = f"${process[:2]} \\to d\\bar{{d}} + {process[-2:]}$"
        if loss_name is not None:
            loss_name = {"heteroschedastic": r"\text{het}", "MSE": r"\text{MSE}"}[
                loss_name
            ]
            process_name = (
                rf"{process_name}"
                if process_name is not None
                else loss_name
            )
        self.losses = losses
        self.metrics = metrics if metrics is not None else {}
        self.process = process
        self.process_name = process_name
        self.regress = regress
        self.regress_name = {
            "r": "r",
            "difference": "\Delta_{\\text{FC} - \\text{LC}}",
            "LC": "A_{\\text{LC}}",
            "FC": "A_{\\text{FC}}",
        }[regress]
        self.debug = debug
        self.model_name = {'MLP': 'MLP', 'Transformer': 'Transformer', 'LGATr': 'L-GATr', 'GNN': 'GNN'}[model_name] if model_name is not None else "NN"

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
                ax.set_yscale("log") if np.all(
                    np.array(self.losses["trn"]) > 0
                ) else ax.set_yscale("symlog")
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

    def plot_weights(
        self,
        file: str,
        split="tst",
        ppd: bool = False,
        percentage_of_ratio_data: float = 100.0,
        pickle_file: Optional[str] = None,
        fix_bins: bool = False,
    ):
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

        split_mode_metrics = self.metrics[split][
            {True: "ppd", False: "raw"}[ppd]
        ]  # select
        compute_and_log_metrics(
            reweight_factors_pred=reweight_factors_pred,
            reweight_factors_truth=reweight_factors_truth,
            split=split,
            ppd=ppd,
            file=file,
            metrics=split_mode_metrics,
        )

        if fix_bins and not ppd:
            self.logger.info(f"         Fixing bins for factors plots, ppd={ppd}")
            percentage_of_ratio_data = -1
            bins_targets = bins_dict[self.regress]["targets"][self.process]
            bins_ratios = bins_dict[self.regress]["ratios"][self.process]
            bins_deltas = bins_dict[self.regress]["deltas"][self.process]
            bins_abs_deltas = bins_dict[self.regress]["abs_deltas"][self.process]
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
                metrics={
                    k: split_mode_metrics[k]
                    for k in ["loss", "eval_time", "eff_2nd_std", "eff_1st_surr"]
                    if k in split_mode_metrics
                },
                bins=bins_targets,
            )
            self._plot_ratios(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                metrics={
                    k: split_mode_metrics[k]
                    for k in ["eff_2nd_surr", "eff_2nd_surr_pm9999", "eff_2nd_surr_pm9995", "eff_2nd_surr_pm999", "eff_2nd_surr_pm995", "eff_2nd_surr_pm99"]
                    if k in split_mode_metrics
                },
                bins=bins_ratios,
            )
            self._plot_deltas(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                metrics={
                    k: split_mode_metrics[k]
                    for k in ["delta_mean", "delta_std"]
                    if k in split_mode_metrics
                },
                bins=bins_deltas,
            )
            self._plot_deltas(
                pp,
                reweight_factors_pred,
                reweight_factors_truth,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file,
                abs=True,
                metrics={
                    k: split_mode_metrics[k]
                    for k in [
                        "abs_delta_mean",
                        "abs_delta_qmin",
                        "abs_delta_qmax",
                        "abs_delta_min",
                        "abs_delta_max",
                    ]
                    if k in split_mode_metrics
                },
                bins=bins_abs_deltas,
            )

    def _plot_targets(
        self,
        pp: PdfPages,
        pred: np.ndarray,
        truth: np.ndarray,
        ppd: bool = False,
        pickle_file: str = None,
        metrics: Optional[Metric] = None,
        bins: Optional[np.ndarray] = None,
    ) -> None:
        """
        Makes a plot of the regressed factors, whichever they are.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """
        label = (
            f"${{{self.regress_name}}}(x)$"
            if not ppd
            else f"$\\tilde{{{self.regress_name}}}(x)$"
        )
        if bins is None:
            xlim_bins = np.array(
                [0.9 * min(min(truth), min(pred)), 1.1 * max(max(truth), max(pred))]
            )
            xlim_bins[0] = 1.1 * xlim_bins[0] if xlim_bins[0] < 0 else xlim_bins[0]
            bins = (
                np.linspace(*xlim_bins, 64)
                if self.regress == "r" or ppd or self.regress == "difference"
                else np.logspace(*np.log10(xlim_bins), 64)
                if np.all(xlim_bins > 0)
                else np.logspace(
                    np.log10(max(1e-10, xlim_bins[0])),
                    np.log10(max(1e-11, xlim_bins[1])),
                    64,
                )
            )
            if self.regress == "r" and ppd:
                bins = np.linspace(
                    *np.percentile(
                        np.concatenate([pred, truth]),
                        [0.005, 99.995],
                    ),
                    64,
                )
                label = f"$\\tilde{{{self.regress_name}}}(x)(99.99\\%)$"

        y_truth, y_truth_err = compute_hist_data(bins, truth, bayesian=False)
        y_pred, y_pred_err = compute_hist_data(bins, pred, bayesian=False)

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
            xlabel=label,
            xscale="linear"
            if self.regress == "difference"
            else "log"
            if not self.regress == "r" and not ppd
            else "linear",
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {"targets-lines": lines, "targets-bins": bins}
            append_to_pickle(pickle_file, pickle_data)

    def _plot_ratios(
        self,
        pp: PdfPages,
        pred: np.ndarray,
        truth: np.ndarray,
        ppd: bool = False,
        percentage_of_ratio_data: float = 100.0,
        pickle_file: str = None,
        eps=1e-20,
        metrics: Optional[Metric] = None,
        bins: Optional[np.ndarray] = None,
    ) -> None:
        """
        Makes a plot of the ratio of the truth and predicted distributions.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """

        ratios = truth / pred if not self.regress == "difference" else truth - pred

        if bins is None:
            xlim_bins = np.percentile(
                ratios,
                [50 - percentage_of_ratio_data / 2, 50 + percentage_of_ratio_data / 2],
            )
            bins = (
                np.linspace(*xlim_bins, 64)
                if self.regress == "r" or ppd or self.regress == "difference"
                else np.logspace(*np.log10(xlim_bins), 64)
                if np.all(xlim_bins > 0)
                else np.logspace(
                    np.log10(max(1e-11, xlim_bins[0])),
                    np.log10(max(1e-11, xlim_bins[1])),
                    64,
                )
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
            label = rf"$\tilde{{{self.regress_name}}}^{{\text{{truth}}}} / \tilde{{{self.regress_name}}}^{{\text{{pred}}}}{perc_str}$"
        else:
            label = rf"${self.regress_name}^{{\text{{truth}}}} / {self.regress_name}^{{\text{{pred}}}}{perc_str}$"

        hist_weights_plot(
            pp,
            lines,
            bins,
            show_ratios=False,
            xlabel=label,
            xscale="linear"
            if self.regress == "r" or ppd or self.regress == "difference"
            else "log",
            title=self.process_name if self.process_name is not None else None,
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {"ratios-lines": lines, "ratios-bins": bins}
            append_to_pickle(pickle_file, pickle_data)

    def _plot_deltas(
        self,
        pp: PdfPages,
        pred: np.ndarray,
        truth: np.ndarray,
        ppd: bool = False,
        percentage_of_ratio_data: float = 100.0,
        pickle_file: str = None,
        eps=1e-20,
        abs=False,
        metrics: Optional[Metric] = None,
        bins: Optional[np.ndarray] = None,
    ) -> None:
        """
        Makes a plot of the delta of the truth and predicted distributions.
        Args:
            pp: PdfPages object
            pred: Predicted values
            truth: True values
            ppd: If True, use preprocessed data
            pickle_file: Path to the output pickle file (optional)
        """

        delta = (pred - truth) / truth
        delta = np.abs(delta) if abs else delta  # take absolute value if abs is True
        # xlim_bins = np.array([0.9*min(delta), 1.1*max(delta)])
        if bins is None:
            xlim_bins = np.percentile(
                delta,
                [50 - percentage_of_ratio_data / 2, 50 + percentage_of_ratio_data / 2],
            )
            if not abs:
                bins = np.linspace(*xlim_bins, 64)
            else:
                bins = np.logspace(*np.log10(np.array(xlim_bins)), 64)

        y_delta, y_delta_err = compute_hist_data(bins, delta, bayesian=False)
        lines = [
            Line(
                y=y_delta,
                y_err=y_delta_err,
                label=f"{self.model_name}",
                color=NN_COLORS[self.model_name],
            ),
        ]
        perc_str = (
            f"({percentage_of_ratio_data:.0f}\\%)"
            if percentage_of_ratio_data >= 0
            else ""
        )
        delta_expr = (
            rf"\tilde{{\Delta}}_{{{self.regress_name}}} = \frac{{\tilde{{{self.regress_name}}}^{{\text{{pred}}}} - \tilde{{{self.regress_name}}}^{{\text{{true}}}}}}{{\tilde{{{self.regress_name}}}^{{\text{{true}}}}}}"
            if ppd
            else rf"\Delta_{{{self.regress_name}}} = \frac{{{self.regress_name}^{{\text{{pred}}}} - {self.regress_name}^{{\text{{true}}}}}}{{{self.regress_name}^{{\text{{true}}}}}}"
        )
        xlabel = f"${delta_expr}{perc_str}$"
        hist_weights_plot(
            pp,
            lines,
            bins,
            show_ratios=False,
            xlabel=xlabel
            if not abs
            else f"$|{{\Delta}}_{{{self.regress_name}}}|{perc_str}$"
            if not ppd
            else f"$|\\tilde{{\Delta}}_{{{self.regress_name}}}|{perc_str}$",
            xscale="log" if abs else "linear",
            title=self.process_name if self.process_name is not None else None,
            no_scale=True,
            metrics=metrics,
            model_name=self.model_name,
        )
        if pickle_file is not None:
            pickle_data = {
                f"deltas{'_abs' if abs else ''}-lines": lines,
                f"deltas{'_abs' if abs else ''}-bins": bins,
            }
            append_to_pickle(pickle_file, pickle_data)

    def plot_amplitudes_closure_test(
        self,
        cfg,
        file: PdfPages,
        split: str = "tst",
        ppd: bool = False,
        pickle_file: str = None,
        metrics: Optional[Metric] = None,
    ) -> None:
        """
        Makes a plot of the closure test of the regression.
        The idea is to plot r_pred * LC amplitude, which should have a relative precision equal to r_pred,
        but matching almost perfectly the FC amplitude.
        Args:
            pp: PdfPages object
            split: Data split to use (e.g., "tst", "trn", "val")
            ppd: Always False, as this method is not implemented for ppd data
            pickle_file: Path to the output pickle file (optional)
            metrics: Metrics to be displayed on the plot (optional)
            bins: Bins to be used for the histogram (optional)
        """
        assert (
            self.regress_name == "r"
        ), "This method is only for the 'r' regression type."
        assert not ppd, "This method is not implemented for ppd data."
        bins = bins_dict["FC"]["targets"][self.process]
        reweight_factors_pred = (
            self.dataset.predicted_factors_raw[split].squeeze().detach().cpu().numpy()
        )
        label = r"$\mathcal{A}_{\text{FC}}$"
        data_path = cfg.dataset.data_path
        type = cfg.dataset.type
        process = cfg.dataset.process
        file_path = os.path.join(data_path, type, paths_dict[process])

        try:
            momenta = np.load(file_path)[..., -3:]
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}. Please check the path.")
            return
        # split the data
        for i, s in enumerate(["trn", "tst", "val"]):
            globals()[f"{s}_slice"] = int(momenta.shape[0] * cfg.dataset.trn_tst_val[i])
        momenta = {
            "trn": momenta[:trn_slice],
            "tst": momenta[trn_slice : trn_slice + tst_slice],
            "val": momenta[-val_slice:],
        }[split]

        FC = momenta[:, -1]
        LC = momenta[:, -3]
        FC_pred = reweight_factors_pred * LC

        with PdfPages(file) as pp:
            y_truth, y_truth_err = compute_hist_data(bins, FC, bayesian=False)
            y_pred, y_pred_err = compute_hist_data(bins, FC_pred, bayesian=False)

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
                    label=rf"$\text{{{self.model_name}}}\ (r^{{\text{{pred}}}} \cdot \mathcal{{A}}_{{\text{{LC}}}})$",
                    color=NN_COLORS[self.model_name],
                ),
            ]
            hist_weights_plot(
                pp,
                lines,
                bins,
                show_ratios=True,
                title=self.process_name if self.process_name is not None else None,
                xlabel=label,
                xscale="log",
                no_scale=True,
                metrics=None,
                model_name=self.model_name,
            )
            if pickle_file is not None:
                pickle_data = {"ampl-targets-lines": lines, "ampl-targets-bins": bins}
                append_to_pickle(pickle_file, pickle_data)

            bins = bins_dict["r"]["ratios"][self.process]
            y_ratio, y_ratio_err = compute_hist_data(bins, FC / FC_pred, bayesian=False)
            lines = [
                Line(
                    y=y_ratio,
                    y_err=y_ratio_err,
                    y_ref=None,
                    label=rf"$\text{{{self.model_name}}}\ (r^{{\text{{pred}}}} \cdot \mathcal{{A}}_{{\text{{LC}}}})$",
                    color=NN_COLORS[self.model_name],
                ),
            ]
            hist_weights_plot(
                pp,
                lines,
                bins,
                show_ratios=False,
                title=self.process_name if self.process_name is not None else None,
                xlabel=r"$\mathcal{A}_{\text{FC}} / (r^{\text{pred}} \cdot \mathcal{A}_{\text{LC}})$",
                xscale="linear",
                no_scale=True,
                metrics=None,
                model_name=self.model_name,
            )
            if pickle_file is not None:
                pickle_data = {"ampl-ratios-lines": lines, "ampl-ratios-bins": bins}
                append_to_pickle(pickle_file, pickle_data)

            bins = bins_dict["r"]["deltas"][self.process]
            deltas = (FC_pred - FC) / FC
            y_deltas, y_deltas_err = compute_hist_data(bins, deltas, bayesian=False)
            lines = [
                Line(
                    y=y_deltas,
                    y_err=y_deltas_err,
                    y_ref=None,
                    label=rf"$\text{{{self.model_name}}}\ (r^{{\text{{pred}}}} \cdot \mathcal{{A}}_{{\text{{LC}}}})$",
                    color=NN_COLORS[self.model_name],
                ),
            ]
            hist_weights_plot(
                pp,
                lines,
                bins,
                show_ratios=False,
                title=self.process_name if self.process_name is not None else None,
                xlabel=r"$\Delta_{\mathcal{A}_{\text{FC}}} = \frac{r^{\text{pred}} \cdot \mathcal{A}_{\text{LC}} - \mathcal{A}_{\text{FC}} }{\mathcal{A}_{\text{FC}}}$",
                xscale="linear",
                no_scale=True,
                metrics=None,
                model_name=self.model_name,
            )
            if pickle_file is not None:
                pickle_data = {"ampl-deltas-lines": lines, "ampl-deltas-bins": bins}
                append_to_pickle(pickle_file, pickle_data)

            bins = bins_dict["r"]["abs_deltas"][self.process]
            y_abs_deltas, y_abs_deltas_err = compute_hist_data(
                bins, np.abs(deltas), bayesian=False
            )
            lines = [
                Line(
                    y=y_abs_deltas,
                    y_err=y_abs_deltas_err,
                    y_ref=None,
                    label=rf"$\text{{{self.model_name}}}\ (r^{{\text{{pred}}}} \cdot \mathcal{{A}}_{{\text{{LC}}}})$",
                    color=NN_COLORS[self.model_name],
                ),
            ]
            hist_weights_plot(
                pp,
                lines,
                bins,
                show_ratios=False,
                title=self.process_name if self.process_name is not None else None,
                xlabel=r"$|\Delta_{\mathcal{A}_{\text{FC}}}|$",
                xscale="log",
                no_scale=True,
                metrics=None,
                model_name=self.model_name,
            )
            if pickle_file is not None:
                pickle_data = {
                    "ampl-deltas_abs-lines": lines,
                    "ampl-deltas_abs-bins": bins,
                }
                append_to_pickle(pickle_file, pickle_data)

    def plot_2d_correlations(
        self,
        file: str,
        split="tst",
        ppd: bool = False,
        percentage_of_ratio_data: float = 100.0,
        pickle_file: Optional[str] = None,
        fix_bins: Optional[bool] = False,
    ):
        if not ppd:
            truth = self.dataset.events[split][:, -1].squeeze().detach().cpu().numpy()
            pred = (
                self.dataset.predicted_factors_raw[split].squeeze().detach().cpu().numpy()
            )
        else:
            truth = self.dataset.events_ppd[split][:, -1].squeeze().detach().cpu().numpy()
            pred = (
                self.dataset.predicted_factors_ppd[split].squeeze().detach().cpu().numpy()
            )
        if fix_bins and not ppd:
            self.logger.info(f"         Fixing bins for 2d unppd plots")
            percentage_of_ratio_data = -1
            bins_targets = bins_dict[self.regress]["targets"][self.process]
            bins_ratios = bins_dict[self.regress]["ratios"][self.process]
            bins_deltas = bins_dict[self.regress]["deltas"][self.process]
            bins_abs_deltas = bins_dict[self.regress]["abs_deltas"][self.process]
        else:
            bins_targets = None
            bins_ratios = None
            bins_deltas = None
            bins_abs_deltas = None
        if percentage_of_ratio_data >= 0:
            perc_str = f"({percentage_of_ratio_data:.0f}\\%)"
        else:
            perc_str = ""

        with PdfPages(file) as pp:
            self._plot_2d(
                pp,
                x=truth / pred,
                y=pred,
                include_diagonal=False,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file + "_vsratio.pkl"
                if pickle_file is not None
                else None,
                bins=[bins_ratios, bins_targets],
                xlabel=f"${self.regress_name}^{{\\text{{truth}}}}/{self.regress_name}^{{\\text{{pred}}}}{perc_str}$"
                if not ppd
                else f"$\\tilde{{{self.regress_name}}}^{{\\text{{truth}}}}/\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}{perc_str}$",
                ylabel=f"${self.regress_name}^{{\\text{{pred}}}}$"
                if not ppd
                else f"$\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}$",
                xscale="linear" if self.regress == "r" or ppd else "log",
                yscale="linear" if self.regress == "r" or ppd else "log",
            )
            self._plot_2d(
                pp,
                x=truth,
                y=pred,
                include_diagonal=True,
                ppd=ppd,
                percentage_of_ratio_data=percentage_of_ratio_data,
                pickle_file=pickle_file + "_vstruth.pkl"
                if pickle_file is not None
                else None,
                bins=[bins_targets, bins_targets],
                xlabel=f"${self.regress_name}^{{\\text{{truth}}}}{perc_str}$"
                if not ppd
                else f"$\\tilde{{{self.regress_name}}}^{{\\text{{truth}}}}{perc_str}$",
                ylabel=f"${self.regress_name}^{{\\text{{pred}}}}$"
                if not ppd
                else f"$\\tilde{{{self.regress_name}}}^{{\\text{{pred}}}}$",
                xscale="linear" if self.regress == "r" or ppd else "log",
                yscale="linear" if self.regress == "r" or ppd else "log",
            )

    def _plot_2d(
        self,
        pp: PdfPages,
        x: np.ndarray,
        y: np.ndarray,
        include_diagonal: bool = False,
        ppd: bool = False,
        percentage_of_ratio_data: float = 100.0,
        pickle_file: str = None,
        metrics: Optional[Metric] = None,
        bins: Optional[np.ndarray] = None,
        xlabel: str = None,
        ylabel: str = None,
        xscale: str = None,
        yscale: str = None,
    ):
        if bins is None or bins[0] is None or bins[1] is None:
            min_val = np.nanmin(y)
            max_val = np.nanmax(y)
            scale_lo = 0.9 if min_val >= 0 else 1.1  # push lower edge outward
            scale_hi = 1.1 if max_val >= 0 else 0.9  # push upper edge outward
            lo = scale_lo * min_val
            hi = scale_hi * max_val
            xlim_bins = [
                np.percentile(
                    x,
                    [
                        50 - percentage_of_ratio_data / 2,
                        50 + percentage_of_ratio_data / 2,
                    ],
                ),
                np.array([lo, hi]),
            ]
            bins = [
                np.linspace(*xlim_bins[0], 64)
                if self.regress == "r" or ppd
                else np.logspace(*np.log10(xlim_bins[0]), 64)
                if np.all(xlim_bins[0] > 0)
                else np.logspace(
                    np.log10(max(1e-11, xlim_bins[0][0])),
                    np.log10(max(1e-11, xlim_bins[0][1])),
                    64,
                ),
                np.linspace(*xlim_bins[1], 64)
                if self.regress == "r" or ppd
                else np.logspace(*np.log10(xlim_bins[1]), 64)
                if np.all(xlim_bins[1] > 0)
                else np.logspace(
                    np.log10(max(1e-11, xlim_bins[1][0])),
                    np.log10(max(1e-11, xlim_bins[1][1])),
                    64,
                ),
            ]
        cmap = plt.get_cmap("viridis")
        cmap.set_bad("white")
        h, x, y = np.histogram2d(x, y, bins=(bins[0], bins[1]))
        h[h == 0] = np.nan
        # h_norm = h / np.nansum(h)
        h_norm = h
        fig, ax = plt.subplots(figsize=(5, 5))
        norm = safe_lognorm(h_norm.T)
        pcm = ax.pcolormesh(
            bins[0],
            bins[1],
            h_norm.T,
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        if include_diagonal:
            ax.plot(
                [
                    max(bins[0][0], bins[1][0]),
                    min(bins[0][-1], bins[1][-1]),
                ],  # limits of x=y within current axes
                [max(bins[0][0], bins[1][0]), min(bins[0][-1], bins[1][-1])],
                linestyle="--",
                color="gray",
                linewidth=2,
            )
        fig.colorbar(pcm, ax=ax)
        fig.suptitle(f"{self.model_name}")
        ax.set_xscale(xscale if xscale is not None else "linear")
        ax.set_yscale(yscale if xscale is not None else "linear")
        ax.set_xlim(bins[0][0], bins[0][-1])
        ax.set_ylim(bins[1][0], bins[1][-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        corner_text(ax, self.process_name, "left", "top")
        fig.savefig(pp, format="pdf", bbox_inches="tight")
        plt.close()
        if pickle_file is not None:
            pickle_data = {
                "h_norm.T": h_norm.T,
                "bins": bins,
                "norm": norm,
            }
            append_to_pickle(pickle_file, pickle_data)

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
                xlim_bins[0] = (
                    1.1 * xlim_bins[0] if xlim_bins[0] < 0 else 0.91 * xlim_bins[0]
                )
                xlim_bins[1] = (
                    0.91 * xlim_bins[1] if xlim_bins[1] < 0 else 1.1 * xlim_bins[1]
                )
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
