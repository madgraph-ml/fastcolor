import torch
import math

from me_wrapper import MG5AmplitudeWrapper


def yij(pi: torch.Tensor, pj: torch.Tensor) -> torch.Tensor:
    pi3 = pi[:, 1:]
    pj3 = pj[:, 1:]
    pimag = torch.sqrt(pi3[:, 0] ** 2 + pi3[:, 1] ** 2 + pi3[:, 2] ** 2)
    pjmag = torch.sqrt(pj3[:, 0] ** 2 + pj3[:, 1] ** 2 + pj3[:, 2] ** 2)
    pipj = torch.einsum("ij,ij->i", pi3, pj3)
    return pipj / pimag / pjmag


def dij(
    sqrts: torch.Tensor, Ei: torch.Tensor, Ej: torch.Tensor, yij: torch.Tensor
) -> torch.Tensor:
    power = 1.5
    ei_pow = (2 * Ei / sqrts) ** power
    ej_pow = (2 * Ej / sqrts) ** power
    d_ij = ei_pow * ej_pow * (1 - yij) ** power
    return d_ij


def Sij_unnormalized(
    pi: torch.Tensor, pj: torch.Tensor, sqrts: torch.Tensor
) -> torch.Tensor:
    Ei = pi[:, 0]
    Ej = pj[:, 0]
    y_ij = yij(pi, pj)
    d_ijm1 = 1 / dij(sqrts, Ei, Ej, y_ij)
    S_ij = d_ijm1 #* Ej**2 / (Ei**2 + Ej**2)
    return S_ij


def Dcal(p: torch.tensor, sqrts: torch.Tensor) -> torch.Tensor:
    # get the energies
    p3 = p[:, 2]
    p4 = p[:, 3]
    p5 = p[:, 4]
    # get the angles
    S_54 = Sij_unnormalized(p5, p4, sqrts)
    S_53 = Sij_unnormalized(p5, p3, sqrts)
    return S_54 + S_53


def Sij(
    pi: torch.Tensor, pj: torch.Tensor, p_all: torch.Tensor, sqrts: torch.Tensor
) -> torch.Tensor:
    S_ij_un = Sij_unnormalized(pi, pj, sqrts)
    norm = Dcal(p_all, sqrts)
    return S_ij_un / norm


def Sigma_ij(
    p: torch.Tensor, i: int, j: int, mg5_api: MG5AmplitudeWrapper
) -> torch.Tensor:
    m2_real = mg5_api.me2(p)
    ptot = p[:, :2].sum(dim=1)
    sqrts = torch.sqrt(
        ptot[:, 0] ** 2 - ptot[:, 1] ** 2 - ptot[:, 2] ** 2 - ptot[:, 3] ** 2
    )
    pi = p[:, i - 1]
    pj = p[:, j - 1]
    S_ij = Sij(pi, pj, p, sqrts)
    y_ij = yij(pi, pj)
    xi_i = 2 * pi[:, 0] / sqrts
    sigma_ij = (1 - y_ij) * (xi_i**2) * m2_real * S_ij
    return sigma_ij
