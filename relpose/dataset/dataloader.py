import torch

from relpose.dataset import Co3dDataset, Co3dv1Dataset


def get_dataloader(
    batch_size=64,
    dataset="co3dv1",
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
            category=category,
            split=split,
            num_images=num_images,
            debug=debug,
        )
    elif dataset in ["co3d", "co3dv2"]:
        dataset = Co3dDataset(
            category=category,
            split=split,
            num_images=num_images,
            debug=debug,
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
