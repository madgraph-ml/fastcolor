import torch
import numpy as np
from madrecolor.utils.lhereader import LHEReader
import madrecolor.utils.physics as phys
from madrecolor.datasets.dataset import Observable, return_obs, get_hardcoded_bins


class gg_6g:
    def __init__(self, params):
        self.params = params
        self.channels = self.params.get("channels", None)
        self.parameterisation = self.params.get("parameterisation", None)

        if "lorentz_products" in self.parameterisation:
            self.channels = self.parameterisation["lorentz_products"]["channels"]
        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.init_observables()

    def lhe_to_array(self, dir: str):
        """LHE to numpy array"""
        reader = LHEReader(dir)
        events = []
        for i, event in enumerate(reader):
            #particles_out = filter(lambda x: x.status == 1, event.particles)
            momenta = []
            for particle in event.particles:
                mom = np.array([particle.energy, particle.px, particle.py, particle.pz])
                momenta.append(mom)
            momenta = np.hstack(momenta)
            # append as last element the LC_to_FC_factor
            momenta = np.append(momenta, event.LC_to_FC_factor)
            events.append(momenta)
        events = np.stack(events)
        return events

    def init_dataset(self):
        n_events = self.params.get("n_events", None)
        process = self.params["process"]
        path = self.params[process]
        events = self.lhe_to_array(path)
        events = events[:n_events] if n_events is not None else events

        events = torch.tensor(events, device=self.device, dtype=torch.float32)
        if process == 'gg_6g':
            if "lorentz_products" in self.parameterisation:
                # compute the lorentz product for all possible 28 pairs
                products = []
                for i in range(8):
                    for j in range(i+1, 8):
                        products.append(phys.LorentzProduct(events[:, i*4:i*4+4], events[:, j*4:j*4+4]))
                products = torch.stack(products, axis=-1)
                events = torch.cat(
                    [
                        products,
                        events[:, -1].unsqueeze(-1),
                    ],
                    axis=1,
                )
                if 28 not in self.channels:
                    # always make sure that the last channel is included and is the target reweighting factor to learn
                    self.channels.append(28)
            else:
                # events = torch.cat(
                #         [
                #             events[:, 0:4],
                #             events[:, 4:8],
                #             events[:, 8:12],
                #             events[:, 12:16],
                #             events[:, 16:20],
                #             events[:, 20:24],
                #             events[:, -1].unsqueeze(-1),
                #         ],
                #         axis=1,
                #     )
                # if 24 not in self.channels:
                #     # always make sure that the last channel is included and is the target reweighting factor to learn
                #     self.channels.append(24)
                raise NotImplementedError("Not implemented yet")
            
        # split the data
        for i, split in enumerate(["trn", "tst", "val"]):
            globals()[f"{split}_slice"] = int(
                events.shape[0] * self.params["trn_tst_val_split"][i]
            )

        self.events = {
            "trn": events[:trn_slice],
            "tst": events[trn_slice : trn_slice + tst_slice],
            "val": events[-val_slice:],
        }


    def apply_preprocessing(self, reverse = False, eps=1e-10):
        
        if not hasattr(self, "events_ppd") and reverse:
                raise ValueError(
                "Cannot reverse preprocess without having preprocessed the data first"
            )
        elif not reverse and not hasattr(self, "events_ppd"):
                self.mean = {}
                self.std = {}
        else:
            pass
        if "lorentz_products" not in self.params["parameterisation"]:
            # # create feature channels
            # pT_channels = [i for i in self.channels if i % 4 == 0 and i < 24]
            # phi_channels = [i for i in self.channels if i % 4 == 1 and i < 24]
            # eta_channels = [i for i in self.channels if i % 4 == 2 and i < 24]
            # mass_channels = [i for i in self.channels if i % 4 == 3 and i < 24]
            # factor_channels = [24]
            
            # # override temporarily
            # pT_channels = []
            # phi_channels = []
            # eta_channels = []
            # mass_channels = []
            raise NotImplementedError("Not implemented yet")
        else:
            pT_channels = []
            phi_channels = []
            eta_channels = []
            mass_channels = [i for i in self.channels if i < 28]
            factor_channels = [28]
        
        if not reverse:
            events_ppd = {
                    f"{k}": self.events[f"{k}"].clone().to(self.device)
                    for k in ["trn", "tst", "val"]
                }
            for split in ["trn", "tst", "val"]:
                # Gaussianize pT, phi, eta, mass
                for pt_ch in pT_channels:
                    events_ppd[split][:, pt_ch] = torch.log(events_ppd[split][:, pt_ch])
                for phi_ch in phi_channels:
                    events_ppd[split][:, phi_ch] = phys.stable_arctanh(
                        events_ppd[split][:, phi_ch] / torch.pi, eps
                    )
                for eta_ch in eta_channels:
                    pass
                for mass_ch in mass_channels:
                    events_ppd[split][:, mass_ch] = torch.log(events_ppd[split][:, mass_ch])
                
                # for ch in factor_channels:
                #     self.min_factor = events_ppd[split][:, ch].min()
                #     events_ppd[split][:, ch] = torch.log(events_ppd[split][:, ch] - self.min_factor + eps)


                # Standardize to N(0, 1)
                self.mean[split] = events_ppd[split].mean(dim=0)
                self.std[split] = events_ppd[split].std(dim=0)

                events_ppd[split] = (events_ppd[split] - self.mean[split]) / (
                    self.std[split] + eps
                )
                assert torch.isfinite(
                    events_ppd[split]
                ).all(), f"{split}: {torch.isnan(events_ppd[split]).sum()} {torch.isinf(events_ppd[split]).sum()}"
            self.events_ppd = events_ppd
            # keep the last channel (reweighting factor) as is
            # for k in ["trn", "tst", "val"]:
            #     self.events_ppd[k][:, -1] = self.events[k][:, -1]
        else:
            # reverse preprocessing for the predicted factors
            self.predicted_factors_raw = {}
            for split in ["trn", "tst", "val"]:
                predicted_factors_raw = self.predicted_factors_ppd[split].clone().to(self.device)
                predicted_factors_raw = predicted_factors_raw * (self.std[split][factor_channels] + eps) + self.mean[split][factor_channels]
                # self.predicted_factors_raw[split] = torch.exp(predicted_factors_raw) + self.min_factor - eps
                self.predicted_factors_raw[split] = predicted_factors_raw


    def init_observables(
        self, n_bins: int = 50
    ) -> list[Observable]:
        self.observables = []
        if "lorentz_products" not in self.params["parameterisation"]:
            raise NotImplementedError("Not implemented yet")
        else:
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 0]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 1]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 2]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 3]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 4]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 5]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 6]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 7]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 8]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 9]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 10]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 11]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 12]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 13]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 14]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 15]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 16]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 17]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 18]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 19]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 20]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 21]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 22]),
                    tex_label=r"p_{g_{5}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 23]),
                    tex_label=r"p_{g_{5}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 24]),
                    tex_label=r"p_{g_{5}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 25]),
                    tex_label=r"p_{g_{6}}\cdot p_{g_{7}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 26]),
                    tex_label=r"p_{g_{6}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 27]),
                    tex_label=r"p_{g_{7}}\cdot p_{g_{8}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )





class gg_4g:
    def __init__(self, params):
        self.params = params
        self.channels = self.params.get("channels", None)
        self.parameterisation = self.params.get("parameterisation", None)

        if "lorentz_products" in self.parameterisation:
            self.channels = self.parameterisation["lorentz_products"]["channels"]
        if self.channels is None:
            raise ValueError("Channels not specified in the dataset parameters")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.init_observables()

    def lhe_to_array(self, dir: str):
        """LHE to numpy array"""
        reader = LHEReader(dir)
        events = []
        for i, event in enumerate(reader):
            #particles_out = filter(lambda x: x.status == 1, event.particles)
            momenta = []
            for particle in event.particles:
                mom = np.array([particle.energy, particle.px, particle.py, particle.pz])
                momenta.append(mom)
            momenta = np.hstack(momenta)
            # append as last element the LC_to_FC_factor
            momenta = np.append(momenta, event.LC_to_FC_factor)
            events.append(momenta)
        events = np.stack(events)
        return events

    def init_dataset(self):
        n_events = self.params.get("n_events", None)
        process = self.params["process"]
        path = self.params[process]
        events = self.lhe_to_array(path)
        events = events[:n_events] if n_events is not None else events

        events = torch.tensor(events, device=self.device, dtype=torch.float32)
        if process == 'gg_4g':
            if "lorentz_products" in self.parameterisation:
                # compute the lorentz product for all possible 15 pairs
                products = []
                for i in range(6):
                    for j in range(i+1, 6):
                        products.append(phys.LorentzProduct(events[:, i*4:i*4+4], events[:, j*4:j*4+4]))
                products = torch.stack(products, axis=-1)
                events = torch.cat(
                    [
                        products,
                        events[:, -1].unsqueeze(-1),
                    ],
                    axis=1,
                )
                if 15 not in self.channels:
                    # always make sure that the last channel is included and is the target reweighting factor to learn
                    self.channels.append(15)
            else:
                events = torch.cat(
                        [
                            events[:, 0:4],
                            events[:, 4:8],
                            events[:, 8:12],
                            events[:, 12:16],
                            events[:, 16:20],
                            events[:, 20:24],
                            events[:, -1].unsqueeze(-1),
                        ],
                        axis=1,
                    )
                if 24 not in self.channels:
                    # always make sure that the last channel is included and is the target reweighting factor to learn
                    self.channels.append(24)
            
        # split the data
        for i, split in enumerate(["trn", "tst", "val"]):
            globals()[f"{split}_slice"] = int(
                events.shape[0] * self.params["trn_tst_val_split"][i]
            )

        self.events = {
            "trn": events[:trn_slice],
            "tst": events[trn_slice : trn_slice + tst_slice],
            "val": events[-val_slice:],
        }


    def apply_preprocessing(self, reverse = False, eps=1e-10):
        
        if not hasattr(self, "events_ppd") and reverse:
                raise ValueError(
                "Cannot reverse preprocess without having preprocessed the data first"
            )
        elif not reverse and not hasattr(self, "events_ppd"):
                self.mean = {}
                self.std = {}
        else:
            pass
        if "lorentz_products" not in self.params["parameterisation"]:
            # create feature channels
            pT_channels = [i for i in self.channels if i % 4 == 0 and i < 24]
            phi_channels = [i for i in self.channels if i % 4 == 1 and i < 24]
            eta_channels = [i for i in self.channels if i % 4 == 2 and i < 24]
            mass_channels = [i for i in self.channels if i % 4 == 3 and i < 24]
            factor_channels = [24]
            
            # override temporarily
            pT_channels = []
            phi_channels = []
            eta_channels = []
            mass_channels = []
        else:
            pT_channels = []
            phi_channels = []
            eta_channels = []
            mass_channels = [i for i in self.channels if i < 15]
            factor_channels = [15]
        

        if not reverse:
            events_ppd = {
                    f"{k}": self.events[f"{k}"].clone().to(self.device)
                    for k in ["trn", "tst", "val"]
                }

            for split in ["trn", "tst", "val"]:
                # Gaussianize pT, phi, eta, mass
                for pt_ch in pT_channels:
                    events_ppd[split][:, pt_ch] = torch.log(events_ppd[split][:, pt_ch])
                for phi_ch in phi_channels:
                    events_ppd[split][:, phi_ch] = phys.stable_arctanh(
                        events_ppd[split][:, phi_ch] / torch.pi, eps
                    )
                for eta_ch in eta_channels:
                    pass
                for mass_ch in mass_channels:
                    events_ppd[split][:, mass_ch] = torch.log(events_ppd[split][:, mass_ch])
                
                # for ch in factor_channels:
                #     self.min_factor = events_ppd[split][:, ch].min()
                #     events_ppd[split][:, ch] = torch.log(events_ppd[split][:, ch] - self.min_factor + eps)



                # Standardize to N(0, 1)
                self.mean[split] = events_ppd[split].mean(dim=0)
                self.std[split] = events_ppd[split].std(dim=0)

                events_ppd[split] = (events_ppd[split] - self.mean[split]) / (
                    self.std[split] + eps
                )

                assert torch.isfinite(
                    events_ppd[split]
                ).all(), f"{split}: {torch.isnan(events_ppd[split]).sum()} {torch.isinf(events_ppd[split]).sum()}"
            self.events_ppd = events_ppd
            # keep the last channel (reweighting factor) as is
            # for k in ["trn", "tst", "val"]:
            #     self.events_ppd[k][:, -1] = self.events[k][:, -1]
        else:
            # reverse preprocessing for the predicted factors
            self.predicted_factors_raw = {}
            for split in ["trn", "tst", "val"]:
                predicted_factors_raw = self.predicted_factors_ppd[split].clone().to(self.device)
                predicted_factors_raw = predicted_factors_raw * (self.std[split][factor_channels] + eps) + self.mean[split][factor_channels]
                # self.predicted_factors_raw[split] = torch.exp(predicted_factors_raw) + self.min_factor - eps
                self.predicted_factors_raw[split] = predicted_factors_raw



    def init_observables(
        self, n_bins: int = 50
    ) -> list[Observable]:
        self.observables = []
        if "lorentz_products" not in self.params["parameterisation"]:
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 0]),
                    tex_label=r"E_{g_{1}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 1]),
                    tex_label=r"p_{x, g_{1}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 2]),
                    tex_label=r"p_{y, g_{1}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 3]),
                    tex_label=r"p_{z, g_{1}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 4]),
                    tex_label=r"E_{g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 5]),
                    tex_label=r"p_{x, g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 6]),
                    tex_label=r"p_{y, g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 7]),
                    tex_label=r"p_{z, g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-1000, upper=0
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 8]),
                    tex_label=r"E_{g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 9]),
                    tex_label=r"p_{x, g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 10]),
                    tex_label=r"p_{y, g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 11]),
                    tex_label=r"p_{z, g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-1000, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 12]),
                    tex_label=r"E_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 13]),
                    tex_label=r"p_{x, g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 14]),
                    tex_label=r"p_{y, g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 15]),
                    tex_label=r"p_{z, g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-1000, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 16]),
                    tex_label=r"E_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 17]),
                    tex_label=r"p_{x, g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 18]),
                    tex_label=r"p_{y, g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 19]),
                    tex_label=r"p_{z, g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-1000, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 20]),
                    tex_label=r"E_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=1000
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 21]),
                    tex_label=r"p_{x, g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 22]),
                    tex_label=r"p_{y, g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-500, upper=500
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 23]),
                    tex_label=r"p_{z, g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=-1000, upper=1000
                    ),
                    yscale="linear",
                )
            )
        else:
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 0]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{2}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 1]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 2]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 3]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 4]),
                    tex_label=r"p_{g_{1}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 5]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{3}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 6]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 7]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 8]),
                    tex_label=r"p_{g_{2}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 9]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{4}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 10]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 11]),
                    tex_label=r"p_{g_{3}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 12]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{5}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 13]),
                    tex_label=r"p_{g_{4}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
            self.observables.append(
                Observable(
                    compute=lambda p: return_obs(p[..., :], p[..., 14]),
                    tex_label=r"p_{g_{5}}\cdot p_{g_{6}}",
                    unit="GeV",
                    bins=lambda obs: get_hardcoded_bins(
                        n_bins=n_bins + 1, lower=0, upper=5e5
                    ),
                    yscale="linear",
                )
            )
