import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
import torchvision
from tqdm.auto import tqdm

from relpose.eval.load_model import get_model, get_eval_dataset
from relpose.utils import get_permutations
from relpose.dataset.co3dv1 import TEST_CATEGORIES


def compute_angular_error(rotation1, rotation2):
    R_rel = rotation2 @ rotation1.T
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation2, rotation1.transpose(0, 2, 1))
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def evaluate_category(
    model, params, category="banana", split="train", num_frames=5, use_pbar=False,
):
    dataset = get_eval_dataset(category=category, split=split, params=params)
    device = next(model.parameters()).device

    permutations = get_permutations(num_frames)
    angular_errors = []
    iterable = tqdm(dataset) if use_pbar else dataset
    for metadata in iterable:
        n = metadata["n"]
        sequence_name = metadata["model_id"]
        key_frames = np.linspace(0, n - 1, num=num_frames, dtype=int)
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"]
        rotations = batch["R"]
        images_permuted = images[permutations]
        rotations_permuted = rotations[permutations]
        rotations_gt = torch.bmm(
            rotations_permuted[:, 1], rotations_permuted[:, 0].transpose(1, 2),
        )
        images1 = images_permuted[:, 0].to(device)
        images2 = images_permuted[:, 1].to(device)

        for i in range(len(permutations)):
            image1 = images1[i]
            image2 = images2[i]
            rotation_gt = rotations_gt[i]

            with torch.no_grad():
                queries, logits = model(
                    images1=image1.unsqueeze(0),
                    images2=image2.unsqueeze(0),
                    recursion_level=4,
                    gt_rotation=rotation_gt.to(device).unsqueeze(0),
                )

            probabilities = torch.softmax(logits, -1)
            probabilities = probabilities[0].detach().cpu().numpy()
            best_prob = probabilities.argmax()
            best_rotation = queries[0].detach().cpu().numpy()[best_prob]
            angular_errors.append(
                compute_angular_error(rotation_gt.numpy(), best_rotation)
            )
    return np.array(angular_errors)


def evaluate_pairwise(
    model=None,
    params=None,
    checkpoint_path=None,
    split="train",
    num_frames=5,
    print_results=True,
    use_pbar=False,
    categories=TEST_CATEGORIES,
):
    if model is None or params is None:
        print(checkpoint_path)
        model, params = get_model(checkpoint_path)

    errors_15 = {}
    errors_30 = {}
    for category in categories:
        angular_errors = evaluate_category(
            model=model,
            params=params,
            category=category,
            split=split,
            num_frames=num_frames,
            use_pbar=use_pbar,
        )
        errors_15[category] = np.mean(angular_errors < 15)
        errors_30[category] = np.mean(angular_errors < 30)

    errors_15["mean"] = np.mean(list(errors_15.values()))
    errors_30["mean"] = np.mean(list(errors_30.values()))
    if print_results:
        print(f"{'Category':>10s}{'<15':6s}{'<30':6s}")
        for category in errors_15.keys():
            print(
                f"{category:>10s}{errors_15[category]:6.02f}{errors_30[category]:6.02f}"
            )
    return errors_15, errors_30
