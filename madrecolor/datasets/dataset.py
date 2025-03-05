import torch
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class Observable:
    """
    Data class for an observable used for plotting
    Args:
        compute: Function that computes the observable value for the given momenta
        tex_label: Observable name in LaTeX for labels in plots
        bins: function that returns tensor with bin boundaries for given observable data
        xscale: X axis scale, "linear" (default) or "log", optional
        yscale: Y axis scale, "linear" (default) or "log", optional
        unit: Unit of the observable or None, if dimensionless, optional
        channel: Channel index for the observable, optional
    """

    compute: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    tex_label: str
    bins: Callable[[torch.Tensor], torch.Tensor]
    xscale: str = "linear"
    yscale: str = "linear"
    unit: Optional[str] = None
    channel: Optional[int] = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["compute"] = None
        d["bins"] = None
        return d


def get_hardcoded_bins(n_bins, lower, upper):
    return torch.linspace(lower, upper, n_bins + 1)


def make_overflow_bin(bins, overflow=torch.inf):
    bins[-1] = overflow
    return bins


def round(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.round(obs)


def return_obs(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return obs


def compute_observables(dataset, split: str = "tst", include_ppd: bool = False):
    if not hasattr(dataset, "observables"):
        raise ValueError(
            "No observables defined for this dataset. Make sure init_observables() has been called."
        )
    observables = []
    obs_list = []
    if include_ppd:
        obs_list_ppd = []
    bins = []
    for obs in dataset.observables:

        o = obs.compute(dataset.events[split])
        if include_ppd:
            o_ppd = obs.compute(dataset.events_ppd[split])

        bin = obs.bins(o)
        observables.append(obs)

        obs_list.append(o.cpu().numpy())
        if include_ppd:
            obs_list_ppd.append(o_ppd.cpu().numpy())
        bins.append(bin.cpu().numpy())

    dataset.observables = observables
    dataset.obs = obs_list
    if include_ppd:
        dataset.obs_ppd = obs_list_ppd
    dataset.bins = bins


class HWW_bins:
    @staticmethod
    def get_Ptl1_bins():
        return torch.Tensor([22, 40, 50, 60, 70, 100, 500])

    @staticmethod
    def get_Ptl2_bins():
        return torch.Tensor([15, 22, 30, 40, 50, 200])

    @staticmethod
    def get_Ptj1_bins():
        return torch.Tensor([30, 90, 120, 160, 220, 700])

    @staticmethod
    def get_Ptj2_bins():
        return torch.Tensor([30, 45, 60, 90, 120, 350])

    @staticmethod
    def get_DeltaPhill_bins():
        return torch.Tensor([0, 0.2, 0.4, 0.6, 0.8, 1.4])

    @staticmethod
    def get_DeltaEtajj_bins():
        return torch.Tensor([0, 2.1, 4.0, 4.375, 5.0, 5.5, 6.25, 9.0])

    @staticmethod
    def get_Mjj_bins():
        return torch.Tensor([450, 700, 950, 1200, 1500, 2200, 6000])

    @staticmethod
    def get_DeltaPhijj_bins():
        return torch.Tensor([-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi])

    @staticmethod
    def get_DeltaEtall_bins(overflow=True):
        bins = torch.Tensor([0, 0.4, 0.6, 0.8, 1.0, 5])
        if overflow:
            bins = make_overflow_bin(bins)
        return bins

    @staticmethod
    def get_Mll_bins(overflow=True):
        bins = torch.Tensor([10, 20, 30, 40, 55, 2000])
        if overflow:
            bins = make_overflow_bin(bins)
        return bins

    @staticmethod
    def get_MT_bins():
        return torch.linspace(0, 250, 11)

    @staticmethod
    def get_PtHiggs_bins():
        return torch.Tensor([0, 80, 120, 160, 260, 850])

    @staticmethod
    def get_Ptll_bins():
        return torch.Tensor([0, 60, 80, 100, 140, 1000])

    @staticmethod
    def get_CosThetaStar_bins():
        return torch.Tensor([0, 0.0625, 0.125, 0.1875, 0.3125, 1])
