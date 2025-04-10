import torch
import numpy as np
import src.utils.physics as phys
from .dataset import Observable, return_obs, get_hardcoded_bins
import os


class gg_ng:
    def __init__(self, logger, cfg):
        self.logger = logger
        self.cfg = cfg
        self.channels = self.cfg.get("channels", None)
        self.parameterisation = self.cfg.get("parameterisation", None)

        if self.parameterisation.naive.use:
            self.channels = self.parameterisation.naive.channels
        elif self.parameterisation.lorentz_products.use:
            self.channels = self.parameterisation.lorentz_products.channels
        else:
            raise ValueError("No parameterisation specified")
        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()

    def init_dataset(self):
        n_events = self.cfg.get("n_events", None)
        data_path = self.cfg.get("data_path", None)
        if self.device == "cpu" and "remote" in data_path:
            self.logger.info(
                f"    Data path {data_path} is remote but you are running on local cpu, changing it to data/"
            )
            data_path = "data"
        if data_path is None:
            raise ValueError("Data path not specified in the parameters")
        type = self.cfg["type"]
        process = self.cfg["process"]
        file_path = (
            os.path.join(data_path, type, self.cfg[process])
            if not self.cfg.use_large_file
            else os.path.join(data_path, type, "large", self.cfg[process])
        )
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        else:
            self.logger.info(f"    Loading data from {file_path}")
            try:
                momenta = (
                    np.load(file_path)[:n_events]
                    if n_events is not None
                    else np.load(file_path)
                )
            except Exception as e:
                raise ValueError(f"Error loading file {file_path}: {e}")
        momenta = torch.tensor(momenta, device=self.device, dtype=torch.float32)
        self.n_particles = momenta.shape[1] // 4
        if self.parameterisation.naive.use:
            events = [momenta[:, i : i + 4] for i in range(0, 4 * self.n_particles, 4)]
            events.append(
                momenta[:, -1].unsqueeze(-1)
            )  # Adding the last column separately
            events = torch.cat(events, axis=1)

        elif self.parameterisation.lorentz_products.use:
            # compute the lorentz products for all possible pairs
            products = []
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    products.append(
                        phys.LorentzProduct(
                            momenta[:, i * 4 : i * 4 + 4], momenta[:, j * 4 : j * 4 + 4]
                        )
                    )
            products = torch.stack(products, axis=-1)
            events = torch.cat(
                [
                    products,
                    momenta[:, -1].unsqueeze(-1),
                ],
                axis=1,
            )
        else:
            raise ValueError("No parameterisation specified")

        if self.channels == []:
            # Use all of channels by default
            self.channels = list(np.arange(events.shape[1]))
        self.last_channel = events.shape[1] - 1
        if self.last_channel not in self.channels:
            # always make sure that the last channel is included and is the target reweighting factor to learn
            self.channels.append(self.last_channel)
        self.events = events

    def split_data(self, events, trn_tst_val):
        # split the data
        for i, split in enumerate(["trn", "tst", "val"]):
            globals()[f"{split}_slice"] = int(events.shape[0] * trn_tst_val[i])
        events_split = {
            "trn": events[:trn_slice],
            "tst": events[trn_slice : trn_slice + tst_slice],
            "val": events[-val_slice:],
        }
        return events_split

    def apply_preprocessing(self, reverse=False, eps=1e-10):
        pp_cfg = self.cfg.preprocessing
        if not hasattr(self, "events_ppd") and reverse:
            raise ValueError(
                "Cannot reverse preprocess without having preprocessed the data first"
            )

        if not reverse:
            events_ppd = self.events.clone().to(self.device)

            if self.cfg.parameterisation.naive.use:
                pt_channels = [
                    i for i in self.channels if i % 4 == 0 and i < self.last_channel
                ]
                phi_channels = [
                    i for i in self.channels if i % 4 == 1 and i < self.last_channel
                ]
                eta_channels = [
                    i for i in self.channels if i % 4 == 2 and i < self.last_channel
                ]
                mass_channels = [
                    i for i in self.channels if i % 4 == 3 and i < self.last_channel
                ]

            elif self.cfg.parameterisation.lorentz_products.use:
                pt_channels = [i for i in self.channels if i < self.last_channel]
                phi_channels = []
                eta_channels = []
                mass_channels = []

            if pp_cfg.gaussianize:
                events_ppd = Gaussianize(
                    events_ppd,
                    pt_channels,
                    phi_channels,
                    eta_channels,
                    mass_channels,
                    eps,
                )

            if pp_cfg.standardize:
                self.mean = events_ppd[:, :-1].mean()
                self.std = events_ppd[:, :-1].std()
                events_ppd[:, :-1] = (events_ppd[:, :-1] - self.mean) / (self.std + eps)

            if pp_cfg.equivariant:
                self.logger.info(
                    f"    Equivariant preprocessing for {self.channels[:-1]}"
                )
                assert (
                    self.cfg.parameterisation.naive.use == True
                ), f"    Equivariant preprocessing only applicable for naive parameterisation, not {[p for p in self.cfg.parameterisation if self.cfg.parameterisation[p].use]}"
                self.std = events_ppd[:, :-1].std()
                events_ppd[:, :-1] = events_ppd[:, :-1] / (self.std + eps)

            if pp_cfg.amplitude.standardize:
                self.ampl_mean = events_ppd[:, -1].mean()
                self.ampl_std = events_ppd[:, -1].std()
                events_ppd[:, -1] = (events_ppd[:, -1] - self.ampl_mean) / (
                    self.ampl_std + eps
                )

            if pp_cfg.amplitude.minmax_scaling:
                # Minmax to [0, 1]
                self.ampl_min = events_ppd[:, -1].min()
                self.ampl_max = events_ppd[:, -1].max()
                events_ppd[:, -1] = (events_ppd[:, -1] - self.ampl_min) / (
                    self.ampl_max - self.ampl_min
                )

            assert torch.isfinite(
                events_ppd
            ).all(), f"{torch.isnan(events_ppd).sum()}, {torch.isinf(events_ppd).sum()}"

            # split data on raw and ppd events
            self.events = self.split_data(self.events, self.cfg.trn_tst_val)
            self.events_ppd = self.split_data(events_ppd, self.cfg.trn_tst_val)
            self.logger.info(
                f"    [Train, Test, Val] events: [{len(self.events_ppd['trn'])}, {len(self.events_ppd['tst'])}, {len(self.events_ppd['val'])}]"
            )
        else:
            # reverse preprocessing for the predicted factors
            self.predicted_factors_raw = {}
            for split in ["trn", "tst", "val"]:
                predicted_factors_raw = (
                    self.predicted_factors_ppd[split].clone().to(self.device)
                )

                if pp_cfg.amplitude.minmax_scaling:
                    predicted_factors_raw = (
                        predicted_factors_raw * (self.ampl_max - self.ampl_min)
                        + self.ampl_min
                    )

                if pp_cfg.amplitude.standardize:
                    predicted_factors_raw = (
                        predicted_factors_raw * (self.ampl_std + eps) + self.ampl_mean
                    )

                self.predicted_factors_raw[split] = predicted_factors_raw

    def init_observables(self, n_bins: int = 50) -> list[Observable]:
        self.observables = []
        if self.parameterisation.naive.use:
            for i in range(self.n_particles):
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i]),
                        tex_label=f"E_{{g_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=0, upper=1000
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 1]),
                        tex_label=f"p_{{x, g_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 2]),
                        tex_label=f"p_{{y, g_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 3]),
                        tex_label=f"p_{{z, g_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
        elif self.parameterisation.lorentz_products.use:
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    idx = i * self.n_particles - (i * (i + 1)) // 2 + (j - i - 1)
                    self.observables.append(
                        Observable(
                            compute=lambda p, idx=idx: return_obs(
                                p[..., :], p[..., idx]
                            ),
                            tex_label=f"p_{{g_{i+1}}}\\cdot p_{{g_{j+1}}}",
                            unit=r"\text{GeV}^{2}",
                            bins=lambda obs: get_hardcoded_bins(
                                n_bins=n_bins + 1, lower=0, upper=5e5
                            ),
                            yscale="linear",
                        )
                    )
        else:
            raise ValueError("No parameterisation specified")


class gg_qqbarng(gg_ng):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_observables(self, n_bins: int = 50) -> list[Observable]:
        self.observables = []
        if self.parameterisation.naive.use:
            for i in range(self.n_particles):
                if 0 <= i < 2 or i >= 4:
                    type = "g"
                elif i == 2:
                    type = "q"
                else:
                    type = "\\bar{q}"

                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i]),
                        tex_label=f"E_{{{type}}}"
                        if 2 <= i <= 3
                        else f"E_{{{type}_{i+1 - 2}}}"
                        if i > 3
                        else f"E_{{{type}_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=0, upper=1000
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 1]),
                        tex_label=f"p_{{x, {type}}}"
                        if 2 <= i <= 3
                        else f"p_{{x, {type}_{i+1 - 2}}}"
                        if i > 3
                        else f"p_{{x, {type}_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 2]),
                        tex_label=f"p_{{y, {type}}}"
                        if 2 <= i <= 3
                        else f"p_{{y, {type}_{i+1 - 2}}}"
                        if i > 3
                        else f"p_{{y, {type}_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p, i=i: return_obs(p[..., :], p[..., 4 * i + 3]),
                        tex_label=f"p_{{z, {type}}}"
                        if 2 <= i <= 3
                        else f"p_{{z, {type}_{i+1 - 2}}}"
                        if i > 3
                        else f"p_{{z, {type}_{i+1}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
        elif self.parameterisation.lorentz_products.use:
            for i in range(self.n_particles):
                if 0 <= i < 2 or i >= 4:
                    type1 = "g"
                else:
                    type1 = "q"
                for j in range(i + 1, self.n_particles):
                    if 0 <= j < 2 or j >= 4:
                        type2 = "g"
                    else:
                        type2 = "q"
                    idx = i * self.n_particles - (i * (i + 1)) // 2 + (j - i - 1)
                    self.observables.append(
                        Observable(
                            compute=lambda p, idx=idx: return_obs(
                                p[..., :], p[..., idx]
                            ),
                            tex_label=f"p_{{{type1}_{i+1}}}\\cdot p_{{{type2}_{j+1}}}",
                            unit=r"\text{GeV}^{2}",
                            bins=lambda obs: get_hardcoded_bins(
                                n_bins=n_bins + 1, lower=0, upper=5e5
                            ),
                            yscale="linear",
                        )
                    )
        else:
            raise ValueError("No parameterisation specified")


def Gaussianize(
    x: torch.Tensor,
    pt_channels: list,
    phi_channels: list,
    eta_channels: list,
    mass_channels: list,
    reverse=False,
    eps: float = 1e-10,
) -> torch.Tensor:
    for ch in pt_channels:
        x[:, ch] = gaussianize_pt(x[:, ch], eps)
    for ch in phi_channels:
        x[:, ch] = gaussianize_phi(x[:, ch], eps)
    for ch in eta_channels:
        x[:, ch] = gaussianize_eta(x[:, ch], eps)
    for ch in mass_channels:
        x[:, ch] = gaussianize_mass(x[:, ch], eps)
    return x


def gaussianize_pt(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return torch.log(x)
    else:
        return torch.exp(x.clip(max=10))


def gaussianize_phi(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return phys.stable_arctanh(x / torch.pi, epsilon)
    else:
        return torch.tanh(x) * torch.pi


def gaussianize_eta(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return x
    else:
        return x


def gaussianize_mass(
    x: torch.Tensor, epsilon: float, reverse: bool = False
) -> torch.Tensor:
    if not reverse:
        return torch.log(x)
    else:
        return torch.exp(x.clip(max=10))
