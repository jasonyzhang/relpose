import torch

from relpose.dataset.co3dv1 import Co3dv1Dataset


def get_dataloader(
    batch_size=64,
    dataset="co3d",
    category=("apple",),
    split="train",
    shuffle=True,
    num_workers=8,
    debug=False,
    num_images=2,
):
    if debug:
        num_workers = 0
    if dataset == "co3dv1":
        dataset = Co3dv1Dataset(
            category=category, split=split, num_images=num_images, debug=debug,
        )
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
