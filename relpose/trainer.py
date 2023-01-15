import argparse
import datetime
import json
import os
import os.path as osp
import shutil
import time
from glob import glob

import cv2
import matplotlib
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from relpose.dataset import get_dataloader
from relpose.dataset.co3dv1 import TEST_CATEGORIES, TRAINING_CATEGORIES
from relpose.eval.eval_pairwise import evaluate_pairwise
from relpose.models import RelPose, RelPoseRegressor, generate_hypotheses
from relpose.utils.geometry import generate_noisy_rotation
from relpose.utils.visualize import unnormalize_image, visualize_so3_probabilities

matplotlib.use("Agg")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", nargs="+", type=str, default=["all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=400_000)
    parser.add_argument("--interval_checkpoint", type=int, default=1000)
    parser.add_argument(
        "--interval_delete_checkpoint",
        type=int,
        default=10000,
        help="Interval to delete old checkpoints.",
    )
    parser.add_argument("--interval_visualize", type=int, default=1000)
    parser.add_argument("--interval_evaluate", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset", type=str, default="co3d")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument(
        "--recursion_level",
        type=int,
        default=3,
        help="Recursion level for healpix sampling. (3 corresponds to 36k)",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="equivolumetric",
        help="Sampling mode can be equivolumetric or random.",
    )
    parser.add_argument("--resume", default="", type=str, help="Path to directory.")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Default: 4 * num_gpus"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--pretrained",
        default="",
        help="Path to pretrained model (to load weights from)",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="If True, freezes the image encoder.",
    )
    parser.add_argument("--direct_regression", action="store_true")
    return parser


def get_permutations(num_images):
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                yield (i, j)


class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.num_iterations = int(args.num_iterations)
        self.lr = args.lr
        self.interval_visualize = args.interval_visualize
        self.interval_checkpoint = args.interval_checkpoint
        self.interval_delete_checkpoint = args.interval_delete_checkpoint
        self.interval_evaluate = args.interval_evaluate
        assert self.interval_delete_checkpoint % self.interval_checkpoint == 0
        self.debug = args.debug

        # Experiment settings:
        self.category = args.category
        self.freeze_encoder = args.freeze_encoder
        self.sampling_mode = args.sampling_mode

        self.iteration = 0
        self.epoch = 0

        num_workers = (
            args.num_gpus * 4 if args.num_workers is None else args.num_workers
        )
        if self.category[0] == "all":
            self.category = TRAINING_CATEGORIES
        print("preparing dataloader")
        self.dataloader = get_dataloader(
            category=self.category,
            dataset=args.dataset,
            split="train",
            batch_size=self.batch_size,
            num_workers=num_workers,
            debug=self.debug,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.direct_regression:
            self.net = RelPoseRegressor(
                num_images=self.num_images, freeze_encoder=self.freeze_encoder,
            )
        else:
            self.net = RelPose(
                num_layers=4,
                num_pe_bases=8,
                hidden_size=256,
                sample_mode=args.sampling_mode,
                recursion_level=args.recursion_level,
                num_queries=36864,  # To match recursion level = 3
                freeze_encoder=self.freeze_encoder,
            )
        self.net = DataParallel(self.net, device_ids=list(range(args.num_gpus)))
        self.net.to(self.device)
        self.start_time = None

        # Setup output directory.
        name = datetime.datetime.now().strftime("%m%d_%H%M")
        if self.debug:
            name += "_debug"
        name += args.name
        name += f"_{args.dataset}"
        if args.dataset == "co3d":
            name += f"_ncat{len(self.category)}"
            if len(self.category) != len(TRAINING_CATEGORIES):
                name += f"{'-'.join(sorted(args.category))}"
        if args.sampling_mode != "equivolumetric":
            name += f"_{args.sampling_mode}"
        if self.batch_size != 64:
            name += f"_b{args.batch_size}"
        if args.direct_regression:
            name += "_direct"
        if args.lr != 0.001:
            name += f"_lr{args.lr}"

        if args.pretrained != "":
            name += "_pre" + osp.basename(args.pretrained)[:9]
        if args.freeze_encoder:
            name += "_freeze"

        # Resume checkpoint.
        if args.resume:
            self.output_dir = args.resume
            self.checkpoint_dir = osp.join(self.output_dir, "checkpoints")
            last_checkpoint = sorted(os.listdir(self.checkpoint_dir))[-1]
            self.load_model(osp.join(self.checkpoint_dir, last_checkpoint))
        else:
            self.output_dir = osp.join(args.output_dir, name)
            self.checkpoint_dir = osp.join(self.output_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            with open(osp.join(self.output_dir, "args.json"), "w") as f:
                json.dump(vars(args), f)
            # Make a copy of the code.
            shutil.copytree("relpose", osp.join(self.output_dir, "relpose"))

            print("Output Directory:", self.output_dir)

        if args.pretrained != "":
            checkpoint_dir = osp.join(args.pretrained, "checkpoints")
            last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
            self.load_model(
                osp.join(checkpoint_dir, last_checkpoint), load_metadata=False
            )

        # Setup tensorboard.
        self.writer = SummaryWriter(log_dir=self.output_dir, flush_secs=10)

    def train(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        permutations = list(get_permutations(self.num_images))  # For N > 2.
        while self.iteration < self.num_iterations:
            for batch in self.dataloader:
                images = batch["image"].to(self.device, non_blocking=True)
                optimizer.zero_grad()
                if self.num_images == 2:
                    image1 = images[:, 0]
                    image2 = images[:, 1]
                    if self.append_image:
                        image3 = images[:, -1]
                    if self.perspective_correction == "coords":
                        bbox1 = batch["bbox_normalized_1"].to(
                            self.device, non_blocking=True
                        )
                        bbox2 = batch["bbox_normalized_2"].to(
                            self.device, non_blocking=True
                        )
                        metadata = torch.cat((bbox1, bbox2), dim=1)
                    else:
                        metadata = None
                    relative_rotation = batch["relative_rotation"].to(
                        self.device, non_blocking=True
                    )
                    if self.eps_noise > 0:
                        noise = generate_noisy_rotation(
                            n=len(relative_rotation),
                            eps=self.eps_noise,
                            degrees=True,
                            device=self.device,
                        )
                        relative_rotation = torch.bmm(relative_rotation, noise)
                    if self.direct_regression:
                        rotation_pred = self.net(images1=image1, images2=image2,)
                        # loss = torch.mean((rotation_pred - relative_rotation) ** 2)
                        # err = (relative_rotation - rotation_pred).norm(dim=(1, 2)) ** 2
                        # loss = torch.mean(err)
                        if self.old_multiply:
                            delta_rot = torch.einsum(
                                "aij,akj->aik", relative_rotation, rotation_pred
                            )
                        else:
                            delta_rot = torch.einsum(
                                "aji,ajk->aik", rotation_pred, relative_rotation
                            )
                        trace = torch.einsum("aii->a", delta_rot)
                        loss = torch.mean(3 - trace)

                        # These are for visualization.
                        # (B, 2, 3, 3)
                        queries = torch.stack([relative_rotation, rotation_pred], dim=1)
                        # (B, 2)
                        logits = torch.tensor([0.0, 1], device=self.device).repeat(5, 1)
                    else:
                        queries, logits = self.net(
                            images1=image1,
                            images2=image2,
                            images3=image3 if self.append_image else None,
                            gt_rotation=relative_rotation,
                            metadata=metadata,
                        )
                        log_prob = torch.log_softmax(logits, dim=-1)
                        if self.soft_loss_threshold > 0:
                            delta_rot = torch.einsum(
                                "abij,akj->abik", queries, relative_rotation
                            )
                            trace = torch.einsum("abii->ab", delta_rot)
                            dist = torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))
                            dist = dist * 180 / np.pi  # Convert to degrees.
                            if self.use_gaussian:
                                sigma = self.soft_loss_threshold
                                weights = torch.exp(-0.5 * (dist / sigma) ** 2)
                                # Zero out far away rotations.
                                weights[dist > 3 * sigma] = 0
                            else:
                                weights = (dist < self.soft_loss_threshold).float()
                            weights_norm = weights / weights.sum(dim=-1, keepdim=True)
                            loss = -torch.sum(weights_norm * log_prob) / self.batch_size
                        else:
                            loss = -torch.mean(log_prob[:, 0])
                else:
                    rotations_gt = batch["R"].to(self.device, non_blocking=True)
                    num_queries = 36864
                    num_queries = 10000
                    hypotheses = generate_hypotheses(  # (B, N_I, N_q, 3, 3)
                        rotations_gt=rotations_gt, num_queries=num_queries,
                    )
                    scores = torch.zeros(
                        (self.batch_size, num_queries), device=self.device
                    )
                    features = [
                        self.net.module.feature_extractor(images[:, i])
                        for i in range(self.num_images)
                    ]
                    if self.num_permutations > len(permutations):
                        perm_ids = np.random.choice(
                            len(permutations), self.num_permutations, replace=False
                        )
                        perm = permutations[perm_ids]
                    else:
                        perm = permutations
                    for i, j in perm:
                        image1 = images[:, i]
                        image2 = images[:, j]
                        R1 = hypotheses[:, i]
                        R2 = hypotheses[:, j]
                        if self.old_multiply:
                            # R2 @ R1.T
                            relative_rotations = torch.einsum("abij,abkj->abik", R2, R1)
                        else:
                            # R1.T @ R2
                            relative_rotations = torch.einsum("abji,abjk->abik", R1, R2)
                        queries, logits = self.net(
                            features1=features[i],
                            features2=features[j],
                            queries=relative_rotations,
                        )
                        scores += logits
                    log_probs = torch.log_softmax(scores, dim=-1)
                    loss = -torch.mean(log_probs[:, 0])

                loss.backward()
                optimizer.step()

                if self.iteration % self.interval_checkpoint == 0:
                    checkpoint_path = osp.join(
                        self.checkpoint_dir, f"ckpt_{self.iteration:09d}.pth"
                    )
                    self.save_model(checkpoint_path)

                if self.iteration % self.interval_visualize == 0:
                    visuals = self.make_visualization(
                        images1=image1,
                        images2=image2,
                        rotations=queries,
                        probabilities=logits.softmax(dim=-1),
                        model_id=batch["model_id"],
                        category=batch["category"],
                        ind1=batch["ind"][:, 0],
                        ind2=batch["ind"][:, 1],
                    )
                    for v, image in enumerate(visuals):
                        self.writer.add_image(
                            f"Visualization/{v}",
                            image,
                            self.iteration,
                            dataformats="HWC",
                        )

                if self.iteration % 20 == 0:
                    if self.start_time is None:
                        self.start_time = time.time()
                    time_elapsed = np.round(time.time() - self.start_time)
                    time_remaining = np.round(
                        (time.time() - self.start_time)
                        / (self.iteration + 1)
                        * (self.num_iterations - self.iteration)
                    )
                    disp = [
                        f"Iter: {self.iteration:d}/{self.num_iterations:d}",
                        f"Epoch: {self.epoch:d}",
                        f"Loss: {loss.item():.3f}",
                        f"Elap: {str(datetime.timedelta(seconds=time_elapsed))}",
                        f"Rem: {str(datetime.timedelta(seconds=time_remaining))}",
                    ]
                    print(", ".join(disp))
                    self.writer.add_scalar("Loss/train", loss.item(), self.iteration)

                self.iteration += 1

                if (
                    not self.direct_regression
                    and self.iteration % self.interval_evaluate == 0
                ):
                    del (
                        images,
                        image1,
                        image2,
                        queries,
                        logits,
                    )
                    errors_15, errors_30 = evaluate_pairwise(
                        self.net,
                        params=vars(self.args),
                        split="test",
                        print_results=True,
                        use_pbar=True,
                        categories=TEST_CATEGORIES,
                        old_split=False,
                    )
                    for k, v in errors_15.items():
                        self.writer.add_scalar(f"Val/{k}@15", v, self.iteration)
                    for k, v in errors_30.items():
                        self.writer.add_scalar(f"Val/{k}@30", v, self.iteration)

                if self.iteration % self.interval_delete_checkpoint == 0:
                    self.clear_old_checkpoints(self.checkpoint_dir)

                if self.iteration >= self.num_iterations + 1:
                    break
            self.epoch += 1

    def save_model(self, path):
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        save_dict = {
            "state_dict": self.net.state_dict(),
            "iteration": self.iteration,
            "epoch": self.epoch,
            "elapsed": elapsed,
        }
        torch.save(save_dict, path)

    def load_model(self, path, load_metadata=True):
        save_dict = torch.load(path)
        if "state_dict" in save_dict:
            self.net.load_state_dict(save_dict["state_dict"])
            if load_metadata:
                self.iteration = save_dict["iteration"]
                self.epoch = save_dict["epoch"]
                if "elapsed" in save_dict:
                    time_elapsed = save_dict["elapsed"]
                    self.start_time = time.time() - time_elapsed
        else:
            self.net.load_state_dict(save_dict)

    def clear_old_checkpoints(self, checkpoint_dir):
        print("Clearing old checkpoints")
        checkpoint_files = glob(osp.join(checkpoint_dir, "ckpt_*.pth"))
        for checkpoint_file in checkpoint_files:
            checkpoint = osp.basename(checkpoint_file)
            checkpoint_iteration = int("".join(filter(str.isdigit, checkpoint)))
            if checkpoint_iteration % self.interval_delete_checkpoint != 0:
                os.remove(checkpoint_file)

    def make_visualization(
        self,
        images1,
        images2,
        rotations,
        probabilities,
        num_vis=5,
        model_id=None,
        category=None,
        ind1=None,
        ind2=None,
    ):
        images1 = images1[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
        images2 = images2[:num_vis].detach().cpu().numpy().transpose(0, 2, 3, 1)
        rotations = rotations[:num_vis].detach().cpu().numpy()
        probabilities = probabilities[:num_vis].detach().cpu().numpy()

        visuals = []
        for i in range(len(images1)):
            image1 = unnormalize_image(cv2.resize(images1[i], (448, 448)))
            image2 = unnormalize_image(cv2.resize(images2[i], (448, 448)))
            so3_vis = visualize_so3_probabilities(
                rotations=rotations[i],
                probabilities=probabilities[i],
                rotations_gt=rotations[i, 0],
                to_image=True,
                display_threshold_probability=1 / len(probabilities[i]),
            )
            full_image = np.vstack((np.hstack((image1, image2)), so3_vis))
            if model_id is not None:
                cv2.putText(full_image, model_id[i], (5, 40), 4, 1, (0, 0, 255))
                cv2.putText(full_image, category[i], (5, 80), 4, 1, (0, 0, 255))
                cv2.putText(full_image, str(int(ind1[i])), (5, 120), 4, 1, (0, 0, 255))
                cv2.putText(
                    full_image, str(int(ind2[i])), (453, 120), 4, 1, (0, 0, 255)
                )
            visuals.append(full_image)
        return visuals


if __name__ == "__main__":
    args = get_parser().parse_args()
    trainer = Trainer(args)
    trainer.train()
