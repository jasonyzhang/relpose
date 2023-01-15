import json
import os
import os.path as osp

import torch

from relpose.models import RelPose
from relpose.dataset import Co3dv1Dataset


def get_model(checkpoint, device="cuda:0"):
    """
    Loads a model from a checkpoint and any associated metadata.
    """
    if ".pth" not in checkpoint:
        checkpoint_dir = osp.join(checkpoint, "checkpoints")
        last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
        print(f"Loading checkpoint {last_checkpoint}")
        checkpoint = osp.join(checkpoint_dir, last_checkpoint)
    pretrained_weights = torch.load(checkpoint, map_location=device)["state_dict"]
    pretrained_weights = {
        k.replace("module.", ""): v for k, v in pretrained_weights.items()
    }
    args_path = osp.join(osp.dirname(osp.dirname(checkpoint)), "args.json")
    if osp.exists(args_path):
        with open() as f:
            args = json.load(f)
        args["output_dir"] = osp.dirname(osp.dirname(checkpoint))
    else:
        args = {}
    relpose = RelPose(sample_mode="equivolumetric", recursion_level=4)
    relpose.to(device)
    relpose.load_state_dict(pretrained_weights)
    relpose.eval()
    return relpose, args


def get_eval_dataset(category, split, dataset="co3dv1"):
    if dataset == "co3dv1":
        dataset = Co3dv1Dataset(category, split, random_aug=False)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return dataset
