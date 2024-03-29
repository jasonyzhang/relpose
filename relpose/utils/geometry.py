import logging

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix


def generate_random_rotations(n=1, device="cpu"):
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    return quaternion_to_matrix(quats)


def generate_superfibonacci(n=1, device="cpu"):
    """
    Samples n rotations equivolumetrically using a Super-Fibonacci Spiral.

    Reference: Marc Alexa, Super-Fibonacci Spirals. CVPR 22.

    Args:
        n (int): Number of rotations to sample.
        device (str): CUDA Device. Defaults to CPU.

    Returns:
        (tensor): Rotations (n, 3, 3).
    """
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    ind = torch.arange(n, device=device)
    s = ind + 0.5
    r = torch.sqrt(s / n)
    R = torch.sqrt(1.0 - s / n)
    alpha = 2 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        1,
    )
    return quaternion_to_matrix(Q).float()


def generate_equivolumetric_grid(recursion_level=3):
    """
    Generates an equivolumetric grid on SO(3). Deprecated in favor of super-fibonacci
    which is more efficient and does not require additional dependencies.

    Uses a Healpix grid on S2 and then tiles 6 * 2 ** recursion level over 2pi.

    Code adapted from https://github.com/google-research/google-research/blob/master/
        implicit_pdf/models.py

    Grid sizes:
        1: 576
        2: 4608
        3: 36864
        4: 294912
        5: 2359296
        n: 72 * 8 ** n

    Args:
        recursion_level: The recursion level of the Healpix grid.

    Returns:
        tensor: rotation matrices (N, 3, 3).
    """
    import healpy

    log = logging.getLogger("healpy")
    log.setLevel(logging.ERROR)  # Supress healpy linking warnings.

    number_per_side = 2**recursion_level
    number_pix = healpy.nside2npix(number_per_side)
    s2_points = healpy.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = torch.tensor(np.stack([*s2_points], 1))

    azimuths = torch.atan2(s2_points[:, 1], s2_points[:, 0])
    # torch doesn't have endpoint=False for linspace yet.
    tilts = torch.tensor(
        np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)
    )
    polars = torch.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        rot_mats = euler_angles_to_matrix(
            torch.stack(
                [azimuths, torch.zeros(number_pix), torch.zeros(number_pix)], 1
            ),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.stack([torch.zeros(number_pix), torch.zeros(number_pix), polars], 1),
            "XYZ",
        )
        rot_mats = rot_mats @ euler_angles_to_matrix(
            torch.tensor([[tilt, 0.0, 0.0]]), "XYZ"
        )
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = torch.cat(grid_rots_mats, 0)
    return grid_rots_mats.float()
