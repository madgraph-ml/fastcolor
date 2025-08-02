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

def random_lorentz_boost(device='cpu', dtype=torch.float64, return_inverse=False):
    """
    Creates a Lorentz boost in a random direction with a given beta.
    Returns the boost matrix and its inverse.
    beta: float, the velocity as a fraction of the speed of light (0 < beta < 1)
    """
    # Random direction and beta
    beta = torch.rand(1, device=device) * 0.8 + 0.1 # beta between 0.1 and 0.9 to avoid numerical instabilities
    theta = torch.acos(2*torch.rand(1, device=device) - 1)
    phi = (2 * torch.pi * torch.rand(1, device=device))
    n = torch.tensor([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], device=device, dtype=dtype)
    # n = torch.tensor([-0.0926, -0.9361,  0.3394]) # always the same direction
    n = n / n.norm() # this is the unitary vector in the boost direction
    n_inv = -n
    gamma = (1.0 / torch.sqrt(1 - beta ** 2)).item()
    beta_vec = beta * n
    beta_vec_inv = beta * n_inv
    boost = torch.eye(4, device=device, dtype=torch.float64)
    boost_inv = torch.eye(4, device=device, dtype=torch.float64)
    boost[0,0] = gamma
    boost[0,1:4] = -gamma * beta_vec
    boost[1:4,0] = -gamma * beta_vec
    boost_inv[0,0] = gamma
    boost_inv[0,1:4] = -gamma * beta_vec_inv
    boost_inv[1:4,0] = -gamma * beta_vec_inv
    for i in range(3):
        for j in range(3):
            boost[i+1, j+1] += (gamma - 1) * (n[i] * n[j])
            boost_inv[i+1, j+1] += (gamma - 1) * (n_inv[i] * n_inv[j])
    return (boost, boost_inv) if return_inverse else boost

def batch_random_lorentz_boost(batch_size, device='cpu', dtype=torch.float64):
    # 1. Vectorized random beta
    beta = torch.rand(batch_size, device=device, dtype=dtype) * 0.8 + 0.1  # [batch]
    # 2. Vectorized random direction
    costheta = 2 * torch.rand(batch_size, device=device, dtype=dtype) - 1   # uniform in [-1,1]
    sintheta = torch.sqrt(1 - costheta ** 2)
    phi = 2 * torch.pi * torch.rand(batch_size, device=device, dtype=dtype)

    n = torch.stack([
        sintheta * torch.cos(phi),
        sintheta * torch.sin(phi),
        costheta
    ], dim=1)   # [batch, 3]
    n = n / n.norm(dim=1, keepdim=True)  # [batch, 3], just in case

    gamma = 1.0 / torch.sqrt(1 - beta ** 2)        # [batch]
    beta_n = (beta.unsqueeze(1) * n)               # [batch, 3]

    # Prepare boost matrices
    boost = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # [batch, 4, 4]
    boost[:, 0, 0] = gamma
    boost[:, 0, 1:4] = -gamma.unsqueeze(1) * beta_n
    boost[:, 1:4, 0] = -gamma.unsqueeze(1) * beta_n

    gamma_m1 = (gamma - 1.0).unsqueeze(1)  # [batch, 1]
    n_expand = n.unsqueeze(2)  # [batch, 3, 1]

    n_expand_T = n.unsqueeze(1)  # [batch, 1, 3]

    nnT = torch.bmm(n_expand, n_expand_T)  # [batch, 3, 3]

    boost[:, 1:4, 1:4] += gamma_m1[:, None] * nnT  # broadcast and add

    return boost  # [batch, 4, 4]

def random_SO3_matrix(device='cpu', dtype=torch.float64):
    """
    Creates a random Lorentz rotation matrix in SO(3,1) with det=+1.
    """
    # Random unit quaternion for uniform SO(3) rotation
    u1, u2, u3 = torch.rand(3, device=device)
    # fix u1, u2, u3
    # u1, u2, u3 = torch.tensor(0.0926, device=device), torch.tensor(0.9361, device=device), torch.tensor(0.3394, device=device)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)
    # Quaternion to rotation matrix (3x3)
    R3d = torch.tensor([
        [1-2*(q3**2+q4**2), 2*(q2*q3 - q1*q4),   2*(q2*q4 + q1*q3)],
        [2*(q2*q3 + q1*q4), 1-2*(q2**2+q4**2),   2*(q3*q4 - q1*q2)],
        [2*(q2*q4 - q1*q3), 2*(q3*q4 + q1*q2),   1-2*(q2**2+q3**2)]
    ], device=device, dtype=dtype)
    R = torch.eye(4, device=device, dtype=torch.float64)
    R[1:4, 1:4] = R3d
    return R

def batch_random_SO3_matrix(batch_size, device='cpu', dtype=torch.float64):
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
    R3d[:,0,0] = 1-2*(q3**2+q4**2)
    R3d[:,0,1] = 2*(q2*q3 - q1*q4)
    R3d[:,0,2] = 2*(q2*q4 + q1*q3)
    R3d[:,1,0] = 2*(q2*q3 + q1*q4)
    R3d[:,1,1] = 1-2*(q2**2+q4**2)
    R3d[:,1,2] = 2*(q3*q4 - q1*q2)
    R3d[:,2,0] = 2*(q2*q4 - q1*q3)
    R3d[:,2,1] = 2*(q3*q4 + q1*q2)
    R3d[:,2,2] = 1-2*(q2**2+q3**2)

    # 3. Embed into 4x4 matrices
    R = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)  # [batch, 4, 4]
    R[:,1:4,1:4] = R3d
    return R


def random_SL4_matrix(device='cpu', dtype=torch.float64):
    """
    Creates a random SL(4) rotation matrix with det=+1.
    """
    R = torch.randn(4, 4, device=device, dtype=torch.float64)
    # R = torch.tensor([
    #     [ 0.9774,  0.7654,  1.2633, -0.2808],
    #     [ 0.6677, -0.3990,  0.4814, -0.6513],
    #     [-0.2007, -0.9910,  1.2286,  0.3808],
    #     [ 1.9933,  0.1832,  0.0070, -1.9268]],
    #     device=device,
    #     dtype=dtype,
    # )
    R = R / torch.abs(torch.det(R))**0.25  # scales det to +-1
    if torch.det(R) < 0:
        R[:, 0] *= -1.  # glip sign to get det=+1
    return R

def batch_random_SL4_matrix(batch_size, device='cpu', dtype=torch.float64):
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

def random_SO2_matrix(device='cpu', dtype=torch.float64):
    """
    Creates a random SO(2) rotation matrix with det=+1 in 4x4
    """
    phi = torch.rand(1, device=device, dtype=dtype) * 2 * torch.pi
    # phi = torch.tensor(torch.pi/4.) # let's do a fixed, 90 degree flip of px and py
    R2d = torch.tensor([
        [torch.cos(phi), -torch.sin(phi)],
        [torch.sin(phi),  torch.cos(phi)]
    ], device=device, dtype=dtype)
    R = torch.eye(4, device=device, dtype=dtype)
    R[1:3, 1:3] = R2d
    return R

def batch_random_SO2_matrix(batch_size, device='cpu', dtype=torch.float64):
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

def random_shear_matrix(device='cpu', dtype=torch.float64):
    """
    Creates a Shear matrix with a non-zero shear in the px, py components
    that preserves det=+1.
    """
    S = torch.eye(4, device=device, dtype=dtype)
    S[1,2] = torch.randn(1, device=device, dtype=dtype).squeeze()  # px mixes with py
    S[1,3] = torch.randn(1, device=device, dtype=dtype).squeeze()  # px mixes with pz
    S[2,3] = torch.randn(1, device=device, dtype=dtype).squeeze()  # py mixes with pz
    # fix shear values
    # S[1, 2] = torch.tensor(-0.0926, device=device, dtype=dtype)
    # S[1, 3] = torch.tensor(-0.9361, device=device, dtype=dtype)
    # S[2, 3] = torch.tensor(0.3394, device=device, dtype=dtype)
    assert abs(torch.det(S).item() - 1.0) < 1e-8, f"Shear matrix does not have det=+1: {torch.det(S).item()}"
    return S

def batch_random_shear_matrix(batch_size, device='cpu', dtype=torch.float64):
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

    assert torch.all(torch.det(S) - 1.0 < 1e-8), f"Shear matrix does not have det=+1: {S[torch.det(S) - 1.0 >= 1e-8]}"
    return S 

def apply_lorentz_boost_to_tensor(x, boost_matrix, boost_inv_matrix=None):
    # x: [..., n_particles * 7], where columns particle_index * 3:7 = E, Px, Py, Pz
    x_new = x.clone()
    n_particles = x.shape[1] // 7
    for i in range(n_particles):
        idx = 7*i
        mom = x[:, idx+3:idx+7] # momenta starts in index 3 bc [pdgid, coloridx, helidx, E, Px, Py, Pz]
        mom_boost = (boost_matrix @ mom.T).T  # [batch, 4]
        # divide by std to match the order of magnitude of the current dataset (has std = 1)
        # mom_boost = mom_boost / mom_boost.std()
        if boost_inv_matrix is not None:
            mom_inv = (boost_inv_matrix @ mom_boost.T).T
            assert torch.allclose(mom, mom_inv, atol=1e-6), "Boost and inverse boost do not match"
        x_new[:, idx+3:idx+7] = mom_boost

    return x_new

def apply_rotation_to_tensor(x, rotation_matrix):
    # x: [batch, n_particles * 7]
    x_new = x.clone()
    n_particles = x.shape[1] // 7
    for i in range(n_particles):
        idx = 7*i
        mom = x[:, idx+3:idx+7]                    # [batch, 4]
        mom_rot = (rotation_matrix @ mom.T).T       # [batch, 4]
        x_new[:, idx+3:idx+7] = mom_rot
    return x_new

def apply_rotation_to_tensor_vectorized(x, rotation_matrices):
    """
    x: [batch_size, n_particles*7]
    rotation_matrices: [batch_size, 4, 4]
    Returns: x_new of same shape, with all [E,Px,Py,Pz] blocks rotated per event.
    """
    x_new = x.clone()
    n_particles = x.shape[1] // 7

    moms = torch.stack(
        [x[:, 7*i+3:7*i+7] for i in range(n_particles)],
        dim=1
    )  # [batch, n_particles, 4]

    # Apply each batch's rotation to all its particles (vectorized!)
    # [batch, n_particles, 4] = bmm([batch, 4, 4], [batch, n_particles, 4, 1]) squeezed
    moms_rot = torch.matmul(rotation_matrices.unsqueeze(1), moms.unsqueeze(-1)).squeeze(-1)
    # [batch, n_particles, 4]

    # Write back the rotated momenta
    for i in range(n_particles):
        x_new[:, 7*i+3:7*i+7] = moms_rot[:, i, :]
    return x_new

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
