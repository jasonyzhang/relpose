"""
The visualization code is adapted from Implicit-PDF (Murphy et. al.)
github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py

Modified so that the rotations are interpretable as yaw (x-axis), pitch (y-axis), and
roll (color).
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from PIL import Image
from pytorch3d import transforms

rc("font", **{"family": "serif", "serif": ["Times New Roman"]})

EYE = np.eye(3)


def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    ax=None,
    fig=None,
    display_threshold_probability=0,
    to_image=True,
    show_color_wheel=True,
    canonical_rotation=EYE,
    gt_size=2500,
    y_offset=-30,
    dpi=400,
):
    """
    Plot a single distribution on SO(3) using the tilt-colored method.

    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
            the marker
        to_image: If True, return a tensor containing the pixels of the finished
            figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
            color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
            rotation', to change the view of the distribution.  It rotates the
            canonical axes so that the view of SO(3) on the plot is different, which
            can help obtain a more informative view.

    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        eulers = transforms.matrix_to_euler_angles(torch.tensor(rotation), "ZXY")
        eulers = eulers.numpy()

        tilt_angle = eulers[0]
        latitude = eulers[1]
        longitude = eulers[2]

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=gt_size,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=4,
        )

    if ax is None:
        fig = plt.figure(figsize=(4, 2), dpi=dpi)
        ax = fig.add_subplot(111, projection="mollweide")
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = transforms.matrix_to_euler_angles(
        torch.tensor(display_rotations), "ZXY"
    )
    eulers_queries = eulers_queries.numpy()

    tilt_angles = eulers_queries[:, 0]
    longitudes = eulers_queries[:, 2]
    latitudes = eulers_queries[:, 1]

    which_to_display = probabilities > display_threshold_probability

    if rotations_gt is not None:
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, "o")
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
            )

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
    )

    yticks = np.array([-60, -30, 0, 30, 60])
    yticks_minor = np.arange(-75, 90, 15)
    ax.set_yticks(yticks_minor * np.pi / 180, minor=True)
    ax.set_yticks(yticks * np.pi / 180, [f"{y}°" for y in yticks], fontsize=14)
    xticks = np.array([-90, 0, 90])
    xticks_minor = np.arange(-150, 180, 30)
    ax.set_xticks(xticks * np.pi / 180, [])
    ax.set_xticks(xticks_minor * np.pi / 180, minor=True)

    for xtick in xticks:
        # Manually set xticks
        x = xtick * np.pi / 180
        y = y_offset * np.pi / 180
        ax.text(x, y, f"{xtick}°", ha="center", va="center", fontsize=14)

    ax.grid(which="minor")
    ax.grid(which="major")

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap, shading="auto")
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels(
            [r"90$\degree$", r"180$\degree$", r"270$\degree$", r"0$\degree$",],
            fontsize=12,
        )
        ax.spines["polar"].set_visible(False)
        plt.text(
            0.5,
            0.5,
            "Roll",
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    if to_image:
        return plot_to_image(fig)
    else:
        return fig


def plot_to_image(fig):
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close(fig)
    return image_from_plot


def antialias(image, level=1):
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        image = Image.fromarray(image)
    for _ in range(level):
        size = np.array(image.size) // 2
        image = image.resize(size, Image.LANCZOS)
    if is_numpy:
        image = np.array(image)
    return image


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)
