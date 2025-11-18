from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch


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

    compute: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
    tex_label: str
    bins: Callable[[torch.Tensor], torch.Tensor]
    xscale: str = "linear"
    yscale: str = "linear"
    unit: str | None = None
    channel: int | None = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["compute"] = None
        d["bins"] = None
        return d


def get_hardcoded_bins(n_bins, lower, upper):
    return torch.linspace(lower, upper, n_bins + 1)


def get_quantile_bins(
    obs: torch.Tensor,
    n_bins: int,
    percentage_of_data_to_show: float,
    xscale: str = "linear",
):
    q_lo = 0.5 - percentage_of_data_to_show / 200
    q_hi = 0.5 + percentage_of_data_to_show / 200
    xlims = torch.quantile(obs, torch.tensor([q_lo, q_hi], device=obs.device))
    if xscale == "log":
        if torch.all(xlims > 0):
            xlims = torch.log10(xlims)
            bins = torch.logspace(xlims[0], xlims[1], n_bins + 1)
        else:
            raise ValueError(
                "Logarithmic scale is not supported for non-positive values."
            )
    else:
        bins = torch.linspace(xlims[0], xlims[1], n_bins + 1)
    return bins


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
