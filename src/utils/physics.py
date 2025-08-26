import torch


minkowski = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float16))


def covariant2(p1: torch.Tensor, p2: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """
    Minkowski inner product of two 4-vectors.
    """
    assert p1.shape == p2.shape and p1.shape[-1] == 4
    assert p1.dtype == p2.dtype

    g = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=p1.dtype, device=p1.device)

    out = torch.sum(p1 * g * p2, dim=-1, keepdim=keepdim)
    return out

def batch_random_lorentz_boost(batch_size, device="cpu", dtype=torch.float64, z_boost=False):
    # 1. Vectorized random beta
    beta = torch.rand(batch_size, device=device, dtype=dtype) * 0.8 + 0.1  # [batch]
    # 2. Vectorized random direction
    costheta = (
        2 * torch.rand(batch_size, device=device, dtype=dtype) - 1
    )  # uniform in [-1,1]
    sintheta = torch.sqrt(1 - costheta**2)
    phi = 2 * torch.pi * torch.rand(batch_size, device=device, dtype=dtype)

    n = torch.stack(
        [
            sintheta * torch.cos(phi),
            sintheta * torch.sin(phi),
            costheta
        ],
        dim=1
    ) if not z_boost else torch.stack(
        [
            torch.zeros(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=dtype),
            costheta
        ],
        dim=1,
    )
    n = n / n.norm(dim=1, keepdim=True)  # [batch, 3], just in case

    gamma = 1.0 / torch.sqrt(1 - beta**2)  # [batch]
    beta_n = beta.unsqueeze(1) * n  # [batch, 3]

    # Prepare boost matrices
    boost = torch.eye(4, device=device, dtype=dtype).repeat(
        batch_size, 1, 1
    )  # [batch, 4, 4]
    boost[:, 0, 0] = gamma
    boost[:, 0, 1:4] = -gamma.unsqueeze(1) * beta_n
    boost[:, 1:4, 0] = -gamma.unsqueeze(1) * beta_n

    gamma_m1 = (gamma - 1.0).unsqueeze(1)  # [batch, 1]
    n_expand = n.unsqueeze(2)  # [batch, 3, 1]

    n_expand_T = n.unsqueeze(1)  # [batch, 1, 3]

    nnT = torch.bmm(n_expand, n_expand_T)  # [batch, 3, 3]

    boost[:, 1:4, 1:4] += gamma_m1[:, None] * nnT  # broadcast and add

    return boost  # [batch, 4, 4]


def batch_random_SO3_matrix(batch_size, device="cpu", dtype=torch.float64):
    """
    Vectorized: returns [batch_size, 4, 4] random SO(3) matrices embedded in SO(3,1)
    """
    # 1. Random quaternions (Shoemake, uniform)
    u1 = torch.rand(batch_size, device=device, dtype=dtype)
    u2 = torch.rand(batch_size, device=device, dtype=dtype)
    u3 = torch.rand(batch_size, device=device, dtype=dtype)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # 2. Build batch of 3x3 rotation matrices
    R3d = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
    R3d[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    R3d[:, 0, 1] = 2 * (q2 * q3 - q1 * q4)
    R3d[:, 0, 2] = 2 * (q2 * q4 + q1 * q3)
    R3d[:, 1, 0] = 2 * (q2 * q3 + q1 * q4)
    R3d[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    R3d[:, 1, 2] = 2 * (q3 * q4 - q1 * q2)
    R3d[:, 2, 0] = 2 * (q2 * q4 - q1 * q3)
    R3d[:, 2, 1] = 2 * (q3 * q4 + q1 * q2)
    R3d[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)

    # 3. Embed into 4x4 matrices
    R = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # [batch, 4, 4]
    R[:, 1:4, 1:4] = R3d
    return R


def batch_random_SL4_matrix(batch_size, device="cpu", dtype=torch.float64):
    """
    Returns [batch_size, 4, 4] random SL(4) matrices (det=+1).
    """
    # 1. Random matrices
    R = torch.randn(batch_size, 4, 4, device=device, dtype=dtype)

    # 2. Scale each so |det| == 1 (det could be negative)
    dets = torch.linalg.det(R)  # [batch]

    # To avoid division by zero (rare), set det=1 for those
    dets_safe = torch.where(dets.abs() > 1e-12, dets, torch.ones_like(dets))
    scales = torch.abs(dets_safe) ** 0.25
    R = R / scales[:, None, None]  # [batch, 4, 4]

    # 3. If det<0, flip first column to make det>0
    sign_flip = (torch.sign(torch.linalg.det(R)) < 0).float()[:, None]
    R[:, 0] = R[:, 0] * (1 - 2 * sign_flip)

    # 4. Sanity check: all dets should now be +1 (up to numerical precision)
    # assert torch.allclose(torch.linalg.det(R), torch.ones(batch_size, device=device, dtype=dtype), atol=1e-8)

    return R  # [batch, 4, 4]


def batch_random_SO2_matrix(batch_size, device="cpu", dtype=torch.float64):
    """
    Returns [batch_size, 4, 4] random SO(2) rotations (in Px,Py) as 4x4 matrices.
    """
    phi = torch.rand(batch_size, device=device, dtype=dtype) * 2 * torch.pi  # [batch]

    c = torch.cos(phi)
    s = torch.sin(phi)

    # Fill in each 4x4 matrix
    R = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # [batch, 4, 4]
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c

    return R  # [batch, 4, 4]


def batch_random_shear_matrix(batch_size, device="cpu", dtype=torch.float64):
    """
    Returns [batch_size, 4, 4] random shear matrices with det=+1.
    Shears in the px, py, pz components only.
    """
    S = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # [batch, 4, 4]

    # Random shears for each event
    s12 = 2 * torch.rand(batch_size, device=device, dtype=dtype) - 1  # px mixes with py
    s13 = 2 * torch.rand(batch_size, device=device, dtype=dtype) - 1  # px mixes with pz
    s23 = 2 * torch.rand(batch_size, device=device, dtype=dtype) - 1  # py mixes with pz

    S[:, 1, 2] = s12
    S[:, 1, 3] = s13
    S[:, 2, 3] = s23

    assert torch.all(
        torch.abs(torch.det(S) - 1.0) < 1e-8
    ), f"Shear matrix does not have det=+1: {S[torch.abs(torch.det(S) - 1.0) >= 1e-8]}"
    return S


def apply_rotation_to_tensor_vectorized(x, rotation_matrices):
    """
    x: [batch_size, n_particles*5]
    rotation_matrices: [batch_size, 4, 4]
    Returns: x_new of same shape, with all [E,Px,Py,Pz] blocks rotated per event.
    """
    x_new = x.clone()
    n_particles = x.shape[1] // 5

    moms = torch.stack(
        [x[:, 5 * i + 1 : 5 * i + 5] for i in range(n_particles)], dim=1
    )  # [batch, n_particles, 4]

    # Apply each batch's rotation to all its particles (vectorized!)
    # [batch, n_particles, 4] = bmm([batch, 4, 4], [batch, n_particles, 4, 1]) squeezed
    moms_rot = torch.matmul(rotation_matrices.unsqueeze(1), moms.unsqueeze(-1)).squeeze(
        -1
    )
    # [batch, n_particles, 4]

    # Write back the rotated momenta
    for i in range(n_particles):
        x_new[:, 5 * i + 1 : 5 * i + 5] = moms_rot[:, i, :]
    return x_new

def apply_Z2_permutation_vectorized(x, block_size=5):
    """
    x: [B, n_particles*block_size] (no target col)
    Swaps either (0<->1) or a random pair in {2..n-1}, with equal probability.
    """
    B, device = x.shape[0], x.device
    n_particles = x.shape[1] // block_size
    assert n_particles >= 2, "x must have at least 2 particles"
    finals = n_particles - 2
    x_blocks = x.view(B, n_particles, block_size)

    # index map per sample
    idx = torch.arange(n_particles, device=device).expand(B, n_particles).clone()
    ar  = torch.arange(B, device=device)

    # choose group: True => initial swap (0<->1), False => finals swap
    choose_init = (torch.rand(B, device=device) < 0.5)

    # default to initial pair
    i_sel = torch.zeros(B, dtype=torch.long, device=device)
    j_sel = torch.ones (B, dtype=torch.long, device=device)

    if finals >= 2:
        # finals pair i!=j in [2..n-1]
        I = torch.randint(2, n_particles, (B,), device=device)
        J = torch.randint(2, n_particles-1, (B,), device=device)
        J = J + (J >= I)

        # pick per sample according to choose_init
        i_sel = torch.where(choose_init, i_sel, I)
        j_sel = torch.where(choose_init, j_sel, J)

    # swap per sample
    tmp = idx[ar, i_sel].clone()
    idx[ar, i_sel] = idx[ar, j_sel]
    idx[ar, j_sel] = tmp

    x_perm = x_blocks.gather(1, idx.unsqueeze(-1).expand(-1, -1, block_size))
    return x_perm.reshape(B, n_particles * block_size)

def delta_eta(
    p: torch.Tensor, eta1: torch.Tensor, eta2: torch.Tensor, abs: bool = True
) -> torch.Tensor:
    deta = eta1 - eta2
    return torch.abs(deta) if abs else deta


def delta_phi(
    p: torch.Tensor, phi1: torch.Tensor, phi2: torch.Tensor, abs: bool = True
) -> torch.Tensor:
    dphi = phi1 - phi2
    dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.abs(dphi) if abs else dphi


def LorentzProduct(p1, p2, base="EPxPyPz"):
    if base == "PtPhiEtaM":
        p1 = PtPhiEtaM_to_EPxPyPz(p1)
        p2 = PtPhiEtaM_to_EPxPyPz(p2)
    elif base == "EPxPyPz":
        pass
    else:
        raise ValueError(f"Base {base} not recognised")
    return p1[..., 0] * p2[..., 0] - torch.sum(p1[..., 1:] * p2[..., 1:], axis=-1)


def PtPhiEtaM_to_EPxPyPz(PtPhiEtaM, cutoff=10):
    if PtPhiEtaM.shape[-1] == 4:
        pt, phi, eta, mass = PtPhiEtaM[:, torch.arange(4)].T
    elif PtPhiEtaM.shape[-1] == 3:
        pt, phi, eta = PtPhiEtaM[:, torch.arange(3)].T
        mass = torch.zeros_like(pt)  # mass is neglected
    else:
        raise ValueError(f"PtPhiEtaM has wrong shape {PtPhiEtaM.shape}")

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(torch.clip(eta, -cutoff, cutoff))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    EPxPyPz = torch.stack((E, px, py, pz), axis=-1)
    assert torch.isfinite(
        EPxPyPz
    ).all(), f"{torch.isnan(EPxPyPz).sum(axis=0)} {torch.isinf(EPxPyPz).sum(axis=0)}"
    return EPxPyPz


def EPxPyPz_to_PtPhiEtaM(EPxPyPz):
    pt = get_pt(EPxPyPz)
    phi = get_phi(EPxPyPz)
    eta = get_eta(EPxPyPz)
    mass = get_mass(EPxPyPz)

    PtPhiEtaM = torch.stack((pt, phi, eta, mass), axis=-1)
    assert torch.isfinite(
        PtPhiEtaM
    ).all(), f"{torch.isnan(PtPhiEtaM).sum(axis=0)} {torch.isinf(PtPhiEtaM).sum(axis=0)}"
    return PtPhiEtaM


def get_pt(particle):
    return torch.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return torch.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle, eps=1e-10):
    # eta = torch.arctanh(particle[...,3] / p_abs) # numerically unstable
    p_abs = torch.sqrt(torch.sum(particle[..., 1:] ** 2, axis=-1))
    eta = 0.5 * (
        torch.log(torch.clip(torch.abs(p_abs + particle[..., 3]), eps, None))
        - torch.log(torch.clip(torch.abs(p_abs - particle[..., 3]), eps, None))
    )
    return eta


def get_mass(particle, eps=1e-6):
    return torch.sqrt(
        torch.clip(
            particle[..., 0] ** 2 - torch.sum(particle[..., 1:] ** 2, axis=-1),
            eps,
            None,
        )
    )


def stable_arctanh(x, eps):
    # numerically stable implementation of arctanh that avoids log(0) issues
    return 0.5 * (
        torch.log(torch.clip(1 + x, min=eps, max=None))
        - torch.log(torch.clip(1 - x, min=eps, max=None))
    )
