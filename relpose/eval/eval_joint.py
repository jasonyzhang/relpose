"""
Evaluation script for relpose.

Mode can be sequential, mst (maximum spanning tree), or coordinate ascent.
Default is uniform spacing. Use --random_order to use random protocol.

Example:
    python -m relpose.eval.eval_joint \
        --checkpoint /path/to/checkpoint \
        --num_frames 5 \
        --use_pbar \
        --dataset co3dv1 \
        --categories_type seen \
        --mode mst

"""

import argparse
import json
import os
import os.path as osp

import ipdb
import numpy as np
import torch
from tqdm.auto import tqdm

from relpose.dataset.co3dv1 import TEST_CATEGORIES, TRAINING_CATEGORIES
from relpose.eval import compute_angular_error_batch, get_eval_dataset, get_model
from relpose.inference.joint_inference import (
    compute_mst,
    run_coordinate_ascent,
    score_hypothesis,
)
from relpose.utils import get_permutations


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--use_pbar", action="store_true")
    parser.add_argument("--dataset", type=str, default="co3dv1")
    parser.add_argument(
        "--categories_type", type=str, default="seen", choices=["seen", "unseen", "all"]
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--mode",
        type=str,
        default="sequential",
        choices=["sequential", "mst", "coord_asc"],
    )
    parser.add_argument(
        "--random_order",
        action="store_true",
        help="If True, uses random order. Else uses uniform spacing.",
    )
    parser.add_argument(
        "--num_queries",
        default=250_000,
        type=int,
        help="Number of queries to use for coordinate ascent.",
    )
    parser.add_argument(
        "--num_iterations",
        default=200,
        type=int,
        help="Number of iterations to use for coordinate ascent.",
    )
    parser.add_argument(
        "--force", action="store_true", help="If True, replaces existing results."
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--skip", type=int, default=1)
    return parser


def evaluate_category_sequential(
    model,
    category="banana",
    split="train",
    num_frames=5,
    use_pbar=False,
    save_dir=None,
    force=False,
    random_order=False,
    dataset="co3dv1",
    **kwargs,
):
    if save_dir is not None:
        r = "-random" if random_order else "-uniform"
        path = osp.join(
            save_dir, f"{category}-{split}-sequential-{num_frames:03d}{r}.json"
        )
        if osp.exists(path) and not force:
            print(f"{path} already exists, skipping")
            with open(path, "r") as f:
                data = json.load(f)
            angular_errors = []
            for d in data.values():
                angular_errors.extend(d["angular_errors"])
            return np.array(angular_errors)
    dataset = get_eval_dataset(category=category, split=split, dataset=dataset)
    device = next(model.parameters()).device
    permutations = get_permutations(num_frames)
    iterable = tqdm(dataset) if use_pbar else dataset
    all_errors = {}
    angular_errors = []
    if random_order:
        order = np.load(osp.join("data", "sequence_order", f"{category}-known.npz"))
    for metadata in iterable:
        n = metadata["n"]
        sequence_name = metadata["model_id"]

        if random_order:
            key_frames = sorted(order[sequence_name][:num_frames])
        else:
            key_frames = np.linspace(
                0, n - 1, num=num_frames, endpoint=False, dtype=int
            )
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"].to(device)
        rotations = batch["R"]
        rotations_pred = [np.eye(3)]
        for i in range(num_frames - 1):
            image1 = images[i]
            image2 = images[i + 1]
            with torch.no_grad():
                queries, logits = model(
                    images1=image1.unsqueeze(0),
                    images2=image2.unsqueeze(0),
                )
            probabilities = torch.softmax(logits, -1)
            probabilities = probabilities[0].detach().cpu().numpy()
            best_prob = probabilities.argmax()
            best_rotation = queries[0].detach().cpu().numpy()[best_prob]
            rotations_pred.append(rotations_pred[-1] @ best_rotation)
        rotations_pred = np.stack(rotations_pred)
        rotations_gt = rotations.numpy()
        permutations = get_permutations(num_frames)
        R_pred_batched = rotations_pred[permutations]
        R_pred_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_pred_batched[:, 0].transpose(0, 2, 1),
            R_pred_batched[:, 1],
        )
        R_gt_batched = rotations_gt[permutations]
        R_gt_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_gt_batched[:, 0].transpose(0, 2, 1),
            R_gt_batched[:, 1],
        )
        errors = compute_angular_error_batch(R_pred_rel, R_gt_rel)
        angular_errors.extend(errors)
        all_errors[sequence_name] = {
            "R_pred": rotations_pred.tolist(),
            "R_gt": rotations_gt.tolist(),
            "angular_errors": errors.tolist(),
        }
    if save_dir is not None:
        with open(path, "w") as f:
            json.dump(all_errors, f)
    return np.array(angular_errors)


def evaluate_category_mst(
    model,
    category="banana",
    split="train",
    num_frames=5,
    use_pbar=False,
    save_dir=None,
    force=False,
    random_order=False,
    dataset="co3dv1",
    **kwargs,
):
    if save_dir is not None:
        r = "-random" if random_order else "-uniform"
        path = osp.join(save_dir, f"{category}-{split}-mst-{num_frames:03d}{r}.json")
        if osp.exists(path) and not force:
            print(f"{path} already exists, skipping")
            with open(path, "r") as f:
                data = json.load(f)
            angular_errors = []
            for d in data.values():
                angular_errors.extend(d["angular_errors"])
            return np.array(angular_errors)
    dataset = get_eval_dataset(category=category, split=split, dataset=dataset)
    device = next(model.parameters()).device
    permutations = get_permutations(num_frames)

    if random_order:
        order = np.load(osp.join("data", "sequence_order", f"{category}-known.npz"))

    iterable = tqdm(dataset) if use_pbar else dataset
    all_errors = {}
    angular_errors = []
    for metadata in iterable:
        n = metadata["n"]
        if num_frames > n:
            continue
        sequence_name = metadata["model_id"]

        best_rotations = np.zeros((num_frames, num_frames, 3, 3))
        best_probs = np.zeros((num_frames, num_frames))
        if random_order:
            key_frames = sorted(order[sequence_name][:num_frames])
        else:
            key_frames = np.linspace(
                0, n - 1, num=num_frames, dtype=int, endpoint=False
            )
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"].to(device)
        rotations = batch["R"]
        rotations_gt = rotations.numpy()

        for i, j in permutations:
            image1 = images[i].unsqueeze(0).to(device)
            image2 = images[j].unsqueeze(0).to(device)
            with torch.no_grad():
                queries, logits = model(
                    images1=image1,
                    images2=image2,
                )
            probabilities = torch.softmax(logits, -1)
            probabilities = probabilities[0].detach().cpu().numpy()
            best_prob = probabilities.max()
            best_rotation = queries[0].detach().cpu().numpy()[probabilities.argmax()]

            best_rotations[i, j] = best_rotation
            best_probs[i, j] = best_prob

        rotations_pred, edges = compute_mst(
            num_frames=num_frames,
            best_probs=best_probs,
            best_rotations=best_rotations,
        )

        R_pred_batched = rotations_pred[permutations]
        R_pred_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_pred_batched[:, 0].transpose(0, 2, 1),
            R_pred_batched[:, 1],
        )
        R_gt_batched = rotations_gt[permutations]
        R_gt_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_gt_batched[:, 0].transpose(0, 2, 1),
            R_gt_batched[:, 1],
        )
        errors = compute_angular_error_batch(R_pred_rel, R_gt_rel)
        angular_errors.extend(errors)
        all_errors[sequence_name] = {
            "R_pred": rotations_pred.tolist(),
            "R_gt": rotations_gt.tolist(),
            "angular_errors": errors.tolist(),
            "edges": edges,
        }
    if save_dir is not None:
        with open(path, "w") as f:
            json.dump(all_errors, f)
    return np.array(angular_errors)


def evaluate_category_coord_asc(
    model,
    category,
    split="train",
    num_iterations=200,
    num_frames=5,
    use_pbar=False,
    save_dir=None,
    force=False,
    dataset="co3dv1",
    num_queries=250_000,
    skip=1,
    index=0,
    random_order=False,
):
    dataset = get_eval_dataset(category=category, split=split, dataset=dataset)
    device = next(model.parameters()).device
    permutations = get_permutations(num_frames)

    if random_order:
        order = np.load(osp.join("data", "sequence_order", f"{category}-known.npz"))

    angular_errors = []
    iterator = np.arange(len(dataset))[index::skip]
    for i in tqdm(iterator):
        metadata = dataset[i]
        n = metadata["n"]
        if num_frames > n:
            continue
        sequence_name = metadata["model_id"]

        r = "-random" if random_order else "-uniform"
        output_file = osp.join(
            save_dir, f"{category}-{sequence_name}-{split}-{num_frames:03d}{r}.json"
        )
        if osp.exists(output_file) and not force:
            with open(output_file) as f:
                data = json.load(f)
            angular_errors.extend(data["errors"])
            continue

        if random_order:
            key_frames = sorted(order[sequence_name][:num_frames])
        else:
            key_frames = np.linspace(0, n - 1, num=num_frames, dtype=int)
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)
        images = batch["image"].to(device)
        features = model.feature_extractor(images)
        rotations = batch["R"]
        rotations_gt = rotations.numpy()

        mst_path = osp.join(
            save_dir, "../mst", f"{category}-{split}-mst-{num_frames:03d}{r}.json"
        )
        with open(mst_path) as f:
            mst_data = json.load(f)

        initial_hypothesis = np.array(mst_data[sequence_name]["R_pred"])
        rotations_pred = run_coordinate_ascent(
            model=model,
            images=images,
            num_frames=num_frames,
            initial_hypothesis=initial_hypothesis,
            num_iterations=num_iterations,
            num_queries=num_queries,
            use_pbar=use_pbar,
        )
        score = score_hypothesis(
            model=model,
            hypothesis=rotations_pred,
            permutations=torch.from_numpy(permutations),
            features=features,
        )
        rotations_pred = rotations_pred.cpu().numpy()
        R_pred_batched = rotations_pred[permutations]
        R_pred_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_pred_batched[:, 0].transpose(0, 2, 1),
            R_pred_batched[:, 1],
        )
        R_gt_batched = rotations_gt[permutations]
        R_gt_rel = np.einsum(
            "Bij,Bjk ->Bik",
            R_gt_batched[:, 0].transpose(0, 2, 1),
            R_gt_batched[:, 1],
        )
        errors = compute_angular_error_batch(R_pred_rel, R_gt_rel)

        output_data = {
            "joint_score": score.item(),
            "errors": errors.tolist(),
            "R_pred": rotations_pred.tolist(),
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f)
        angular_errors.extend(errors)
    return np.array(angular_errors)


def evaluate_joint(
    model=None,
    checkpoint_path=None,
    dataset="co3dv1",
    categories_type="seen",
    split="test",
    num_frames=5,
    print_results=True,
    use_pbar=False,
    mode="sequential",
    save_output=True,
    force=False,
    num_queries=250_000,
    num_iterations=200,
    random_order=False,
    reverse=False,
    index=0,
    skip=1,
):
    if model is None or params is None:
        print(checkpoint_path)
        model, params = get_model(checkpoint_path)

    if save_output:
        if ".pth" in checkpoint_path:
            model_dir = osp.dirname(osp.dirname(checkpoint_path))
        else:
            model_dir = checkpoint_path
        save_dir = osp.join(model_dir, "eval", mode)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    eval_map = {
        "sequential": evaluate_category_sequential,
        "mst": evaluate_category_mst,
        "coord_asc": evaluate_category_coord_asc,
    }
    eval_function = eval_map[mode]

    errors_15 = {}
    errors_30 = {}

    if categories_type == "seen":
        categories = TRAINING_CATEGORIES
    elif categories_type == "unseen":
        categories = TEST_CATEGORIES
    elif categories_type == "all":
        categories = TRAINING_CATEGORIES + TEST_CATEGORIES
    else:
        raise Exception(f"Unknown categories type: {categories_type}")
    categories = categories[index::skip]
    if reverse:
        categories = categories[::-1]
    for category in categories:
        angular_errors = eval_function(
            model=model,
            dataset=dataset,
            category=category,
            split=split,
            num_frames=num_frames,
            num_iterations=num_iterations,
            use_pbar=use_pbar,
            save_dir=save_dir,
            force=force,
            num_queries=num_queries,
            random_order=random_order,
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    evaluate_joint(
        checkpoint_path=args.checkpoint,
        num_frames=args.num_frames,
        mode=args.mode,
        print_results=True,
        use_pbar=args.use_pbar,
        force=args.force,
        split=args.split,
        dataset=args.dataset,
        num_queries=args.num_queries,
        num_iterations=args.num_iterations,
        random_order=args.random_order,
        reverse=args.reverse,
        index=args.index,
        skip=args.skip,
        categories_type=args.categories_type,
    )
