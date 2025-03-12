import torch
import numpy as np
from madrecolor.utils.lhereader import LHEReader
import madrecolor.utils.physics as phys
from madrecolor.datasets.dataset import Observable, return_obs, get_hardcoded_bins


class gg_ng:
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
        """
        LHE to numpy array reader
        """
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
        momenta = self.lhe_to_array(path)
        momenta = momenta[:n_events] if n_events is not None else momenta
        momenta = torch.tensor(momenta, device=self.device, dtype=torch.float32)
        self.n_particles = momenta.shape[1] // 4
 
        if "lorentz_products" in self.parameterisation:
            self.last_channel = int(self.n_particles * (self.n_particles - 1) / 2)
            # compute the lorentz product for all possible 28 pairs
            products = []
            for i in range(self.n_particles):
                for j in range(i+1, self.n_particles):
                    products.append(phys.LorentzProduct(momenta[:, i*4:i*4+4], momenta[:, j*4:j*4+4]))
            products = torch.stack(products, axis=-1)
            events = torch.cat(
                [
                    products,
                    momenta[:, -1].unsqueeze(-1),
                ],
                axis=1,
            )
        else:
            self.last_channel = 4 * self.n_particles
            event_splits = [events[:, i : i + 4] for i in range(0, 4 * self.n_particles, 4)]
            event_splits.append(events[:, -1].unsqueeze(-1))  # Adding the last column separately
            events = torch.cat(event_splits, axis=1)
        
        if self.last_channel not in self.channels:
            # always make sure that the last channel is included and is the target reweighting factor to learn
            self.channels.append(self.last_channel)

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
            raise NotImplementedError("Not implemented yet")
        else:
            pT_channels = []
            phi_channels = []
            eta_channels = []
            mass_channels = [i for i in self.channels if i<self.last_channel]
            factor_channels = [self.last_channel]
        
        if not reverse:
            events_ppd = {
                    f"{k}": self.events[f"{k}"].clone().to(self.device)
                    for k in ["trn", "tst", "val"]
                }
            for split in ["trn", "tst", "val"]:
                
                # Gaussianize
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
                for ch in factor_channels:
                    pass

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
        else:
            # reverse preprocessing for the predicted factors
            self.predicted_factors_raw = {}
            for split in ["trn", "tst", "val"]:
                predicted_factors_raw = self.predicted_factors_ppd[split].clone().to(self.device)
                predicted_factors_raw = predicted_factors_raw * (self.std[split][factor_channels] + eps) + self.mean[split][factor_channels]
                self.predicted_factors_raw[split] = predicted_factors_raw


    def init_observables(
        self, n_bins: int = 50
    ) -> list[Observable]:
        self.observables = []
        if "lorentz_products" in self.params["parameterisation"]:
            for i in range(self.n_particles):
                for j in range(i+1, self.n_particles):
                    idx = i * self.n_particles - (i * (i + 1)) // 2 + (j - i - 1)
                    self.observables.append(
                        Observable(
                            compute=lambda p, idx=idx: return_obs(p[..., :], p[..., idx]),
                            tex_label=f"p_{{g_{i+1}}}\\cdot p_{{g_{j+1}}}",
                            unit=r"\text{GeV}^{2}",
                            bins=lambda obs: get_hardcoded_bins(
                                n_bins=n_bins + 1, lower=0, upper=5e5
                            ),
                            yscale="linear",
                        )
                    )
        else:
            for i in range(self.n_particles):
                self.observables.append(
                    Observable(
                        compute=lambda p: return_obs(p[..., :], p[..., 4*i]),
                        tex_label=f"E_{{g_{i}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=0, upper=1000
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p: return_obs(p[..., :], p[..., 4*i+1]),
                        tex_label=f"p_{{x, g_{i}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p: return_obs(p[..., :], p[..., 4*i+2]),
                        tex_label=f"p_{{y, g_{i}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=-500, upper=500
                        ),
                        yscale="linear",
                    )
                )
                self.observables.append(
                    Observable(
                        compute=lambda p: return_obs(p[..., :], p[..., 4*i+3]),
                        tex_label=f"p_{{z, g_{i}}}",
                        unit="GeV",
                        bins=lambda obs: get_hardcoded_bins(
                            n_bins=n_bins + 1, lower=0, upper=1000
                        ),
                        yscale="linear",
                    )
                )