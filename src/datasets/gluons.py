import torch
import numpy as np
import src.utils.physics as phys
from .dataset import Observable, return_obs, get_hardcoded_bins, get_quantile_bins
from src.utils.paths_dict import paths as paths_dict
import os


class gg_ng:
    def __init__(self, logger, cfg):
        self.logger = logger
        self.cfg = cfg
        self.channels = self.cfg.get("channels", None)
        self.parameterization = self.cfg.get("parameterization", None)
        self.logger.info(f"    Regressing {self.cfg.get('regress', 'r')}")
        self.regress_target = self.cfg.get("regress", "r")
        self.remove_color = self.cfg.get("remove_color", False)
        self.features_per_particle = 7 if not self.remove_color else 6
        if self.parameterization.naive.use:
            self.channels = self.parameterization.naive.channels
        elif self.parameterization.lorentz_products.use:
            self.channels = self.parameterization.lorentz_products.channels
        else:
            raise ValueError("No parameterization specified")
        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()

    def init_dataset(self):
        n_data = self.cfg.get("n_data", None)
        data_path = self.cfg.get("data_path", None)
        if self.device == "cpu" and "remote" in data_path:
            self.logger.info(
                f"    Data path {data_path} is remote but you are running on local cpu, changing it to data/"
            )
            data_path = "data"
        if data_path is None:
            raise ValueError("Data path not specified in the parameters")
        type = self.cfg.type
        process = self.cfg.process
        #
        file_path = os.path.join(data_path, type, paths_dict[process])

        try:
            momenta = (
                np.load(file_path)[:n_data] if n_data is not None else np.load(file_path)
            )
            if self.regress_target == "r":
                momenta = np.concatenate(
                    [momenta[:, :-3], momenta[:, -1:] / momenta[:, -3:-2]], axis=1
                )
            elif self.regress_target == "difference":
                momenta = np.concatenate(
                    [momenta[:, :-3], momenta[:, -3:-2] - momenta[:, -1:]], axis=1
                )
            elif self.regress_target == "FC":
                momenta = np.concatenate([momenta[:, :-3], momenta[:, -1:]], axis=1)
            elif self.regress_target == "LC":
                momenta = np.concatenate([momenta[:, :-3], momenta[:, -3:-2]], axis=1)
            else:
                raise ValueError(f"Unknown regression target {self.regress_target}")
            self.logger.info(f"    Loaded data from {file_path}, shape {momenta.shape}")
        except FileNotFoundError:
            self.logger.error(
                f"File {file_path} not found. Please check the data path and process name."
            )
            raise

        self.n_particles = (momenta.shape[1] - 1) // 7
        if self.remove_color:
            target = momenta[:, -1:]
            momenta = momenta[:, :-1]
            momenta = momenta.reshape(-1, self.n_particles, 7)
            momenta = np.delete(momenta, [1], axis=-1)
            momenta = momenta.reshape(-1, 6 * self.n_particles)
            momenta = np.concatenate([momenta, target], axis=1)
            self.logger.info(f"    Removed color channel.")
        self.logger.info(
            f"    Number of particles: {self.n_particles}, per-particle features: {(momenta.shape[1] - 1) / self.n_particles}"
        )

        if self.parameterization.naive.use:
            momenta = torch.tensor(momenta, device=self.device, dtype=torch.float64)
            events = momenta[:, :-1]
            self.input_channels = torch.arange(
                self.features_per_particle * self.n_particles,
                dtype=torch.int16,
                device=self.device,
            ).tolist()
        elif self.parameterization.lorentz_products.use:
            target = momenta[:, -1:]  # the last column is the regression factor
            momenta = momenta[:, :-1]  # remove the regression factor from the momenta
            reshaped_momenta = momenta.reshape(
                -1, self.n_particles, self.features_per_particle
            )

            pdg_ids = reshaped_momenta[:, :, 0]
            colors = reshaped_momenta[:, :, 1] if not self.remove_color else None
            helicities = reshaped_momenta[:, :, 2 - int(self.remove_color)]
            momenta = reshaped_momenta[:, :, 3 - int(self.remove_color) :].reshape(
                -1, 4 * self.n_particles
            )
            momenta = np.concatenate([momenta, target], axis=1)
            momenta = torch.tensor(momenta, device=self.device, dtype=torch.float64)

            if self.cfg.embed_helicities.use:
                config_dict = {
                    tuple(cfg): idx
                    for idx, cfg in enumerate(np.unique(helicities, axis=0))
                }
                config_ids = np.array([config_dict[tuple(cfg)] for cfg in helicities])
            # compute the lorentz products for all possible pairs
            base = []
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    base.append(
                        phys.LorentzProduct(
                            momenta[:, i * 4 : i * 4 + 4], momenta[:, j * 4 : j * 4 + 4]
                        )
                    )
            self.input_channels = torch.arange(
                self.n_particles * (self.n_particles - 1) // 2,
                dtype=torch.int16,
                device=self.device,
            ).tolist()
            events = torch.stack(base, dim=1).to(dtype=torch.float64, device=self.device)
        else:
            raise ValueError("No parameterization specified")

        events = torch.cat(
            [
                events,
                momenta[:, -1:],
            ],
            dim=1,
        )
        if self.cfg.embed_helicities.use:
            try:
                events = torch.cat(
                    [
                        events[..., :-1],
                        torch.tensor(
                            config_ids, device=self.device, dtype=torch.int16
                        ).unsqueeze(1),
                        events[..., -1:],
                    ],
                    dim=1,
                ).to(dtype=torch.float64)
                self.helicity_dict_size = len(np.unique(config_ids))
                self.logger.info(
                    f"    Using helicity LUT with {self.helicity_dict_size} configurations at channel {events.shape[1] - 2}"
                )
            except Exception as e:
                self.logger.error(f"Error embedding helicities: {e}")
                raise
            helicity_channel = events.shape[1] - 2
            self.input_channels.append(events.shape[1] - 2)

        if self.channels == []:
            # Use all of channels by default
            self.input_channels = [i for i in range(events.shape[1])]
            if self.cfg.embed_helicities.use:
                self.channels_to_preprocess = [
                    i for i in self.input_channels if i != helicity_channel
                ]
            else:
                self.channels_to_preprocess = [
                    i
                    for i in self.input_channels
                    if i % self.features_per_particle
                    in [x - int(self.remove_color) for x in [3, 4, 5, 6]]
                ]

        self.events = events
        self.logger.info(
            f"    Using channels {self.input_channels[:-1]} for regression, preprocessing {self.channels_to_preprocess}, with channel {self.input_channels[-1]} ({self.cfg.get('regress', 'r')}) as the target"
        )

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

    def apply_preprocessing(self, reverse=False, eps=1e-15, ppd=None):
        pp_cfg = self.cfg.preprocessing
        if not hasattr(self, "events_ppd") and reverse:
            raise ValueError(
                "Cannot reverse preprocess without having preprocessed the data first"
            )

        if not reverse:
            events_ppd = self.events.clone().to(self.device)

            if self.cfg.parameterization.naive.use:
                pt_channels = []
                phi_channels = []
                eta_channels = []
                mass_channels = []

            elif self.cfg.parameterization.lorentz_products.use:
                pt_channels = [
                    i
                    for i in range(int(self.n_particles * (self.n_particles - 1) / 2))
                    if i in self.channels_to_preprocess
                ]
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

            if pp_cfg.equivariant:
                self.logger.info(
                    f"    Equivariant preprocessing for {self.channels_to_preprocess}"
                )
                assert (
                    self.cfg.parameterization.naive.use == True
                ), f"    Equivariant preprocessing only applicable for naive parameterization, not {[p for p in self.cfg.parameterization if self.cfg.parameterization[p].use]}"
                self.std = events_ppd[:, self.channels_to_preprocess].std()
                events_ppd[:, self.channels_to_preprocess] = events_ppd[
                    :, self.channels_to_preprocess
                ] / (self.std + eps)

            if pp_cfg.standardize:
                self.mean = events_ppd[:, self.channels_to_preprocess].mean()
                self.std = events_ppd[:, self.channels_to_preprocess].std()
                events_ppd[:, self.channels_to_preprocess] = (
                    events_ppd[:, self.channels_to_preprocess] - self.mean
                ) / (self.std + eps)

            if pp_cfg.amplitude.log:
                self.logger.info(
                    f"    Log preprocessing for channel {events_ppd.shape[1] - 1}"
                )
                events_ppd[:, -1] = torch.log(events_ppd[:, -1] + eps)

            if pp_cfg.amplitude.minmax_scaling:
                # Minmax to [0, 1]
                self.logger.info(f"    MinMax scaling for {events_ppd.shape[1] - 1}")
                self.ampl_min = events_ppd[:, -1].min()
                self.ampl_max = events_ppd[:, -1].max()
                events_ppd[:, -1] = (events_ppd[:, -1] - self.ampl_min) / (
                    self.ampl_max - self.ampl_min
                )
                events_ppd[:, -1] *= 10
                events_ppd[:, -1] -= 5

            if pp_cfg.amplitude.arctanh:
                self.logger.info(
                    f"    Arctanh preprocessing for channel {events_ppd.shape[1] - 1}"
                )
                self.ampl_min = events_ppd[:, -1].min()
                self.ampl_max = events_ppd[:, -1].max()
                arg = (
                    2
                    * (1 - 1e-16)
                    * (
                        (events_ppd[:, -1] - self.ampl_min)
                        / (self.ampl_max - self.ampl_min)
                        - 1 / 2
                    )
                )
                events_ppd[:, -1] = torch.arctanh(arg)

            if pp_cfg.amplitude.standardize:
                self.logger.info(
                    f"    Standard preprocessing for {events_ppd.shape[1] - 1}"
                )
                self.ampl_mean = events_ppd[:, -1].mean()
                self.ampl_std = events_ppd[:, -1].std()
                events_ppd[:, -1] = (events_ppd[:, -1] - self.ampl_mean) / (
                    self.ampl_std + eps
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
            if ppd is None:
                raise ValueError(
                    "Cannot reverse preprocess without having preprocessed the data first"
                )
            predicted_factors_raw = ppd.clone().to(self.device)

            if pp_cfg.amplitude.standardize:
                predicted_factors_raw = (
                    predicted_factors_raw * (self.ampl_std + eps) + self.ampl_mean
                )

            if pp_cfg.amplitude.arctanh:
                predicted_factors_raw = (
                    torch.tanh(predicted_factors_raw) / (2 * (1 - 1e-16)) + 0.5
                ) * (self.ampl_max - self.ampl_min) + self.ampl_min

            if pp_cfg.amplitude.minmax_scaling:
                predicted_factors_raw += 5
                predicted_factors_raw /= 10
                predicted_factors_raw = (
                    predicted_factors_raw * (self.ampl_max - self.ampl_min)
                    + self.ampl_min
                )

            if pp_cfg.amplitude.log:
                predicted_factors_raw = torch.exp(predicted_factors_raw) - eps

            return predicted_factors_raw

    def init_observables(self, n_bins: int = 50) -> list[Observable]:
        self.observables = []
        if self.parameterization.naive.use:
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
        elif self.parameterization.lorentz_products.use:
            for i in range(self.n_particles):
                for j in range(i + 1, self.n_particles):
                    idx = i * self.n_particles - (i * (i + 1)) // 2 + (j - i - 1)
                    self.observables.append(
                        Observable(
                            compute=lambda p, idx=idx: return_obs(p[..., :], p[..., idx]),
                            tex_label=f"p_{{g_{i+1}}}\\cdot p_{{g_{j+1}}}",
                            unit=r"\text{GeV}^{2}",
                            bins=lambda p, idx=idx: get_quantile_bins(
                                obs=p,
                                n_bins=n_bins,
                                percentage_of_data_to_show=99.0,
                                xscale="log",
                            ),
                            xscale="log",
                            yscale="linear",
                        )
                    )
        else:
            raise ValueError("No parameterization specified")


class gg_ddbarng(gg_ng):
    def __init__(self, logger, cfg):
        super().__init__(logger, cfg)

    def init_observables(self, n_bins: int = 50) -> list[Observable]:
        self.observables = []
        if self.parameterization.naive.use:
            for i in range(self.n_particles):
                if 0 <= i < 2 or i >= 4:
                    type = "g"
                elif i == 2:
                    type = "d"
                else:
                    type = "\\bar{d}"

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
        elif self.parameterization.lorentz_products.use:
            for i in range(self.n_particles):
                if 0 <= i < 2 or i >= 4:
                    type1 = "g"
                else:
                    type1 = "q"
                for j in range(i + 1, self.n_particles):
                    if 0 <= j < 2 or j >= 4:
                        type2 = "g"
                    else:
                        type2 = "d"
                    idx = i * self.n_particles - (i * (i + 1)) // 2 + (j - i - 1)
                    self.observables.append(
                        Observable(
                            compute=lambda p, idx=idx: return_obs(p[..., :], p[..., idx]),
                            tex_label=f"p_{{{type1}_{i+1}}}\\cdot p_{{{type2}_{j+1}}}",
                            unit=r"\text{GeV}^{2}",
                            bins=lambda p, idx=idx: get_quantile_bins(
                                obs=p,
                                n_bins=n_bins,
                                percentage_of_data_to_show=99.0,
                                xscale="log",
                            ),
                            yscale="linear",
                        )
                    )
        else:
            raise ValueError("No parameterization specified")


def Gaussianize(
    x: torch.Tensor,
    pt_channels: list,
    phi_channels: list,
    eta_channels: list,
    mass_channels: list,
    reverse=False,
    eps: float = 1e-100,
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
