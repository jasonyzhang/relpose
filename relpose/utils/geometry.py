import logging

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_to_matrix


def generate_random_rotations(n=1, device="cpu"):
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    return quaternion_to_matrix(quats)


def generate_superfibonacci(n=1):
    pass


def generate_equivolumetric_grid(recursion_level=3):
    """
    Generates an equivolumetric grid on SO(3).

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
