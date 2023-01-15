"""
CO3Dv1 dataset.
"""

import gzip
import json
import os.path as osp

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from relpose.utils.bbox import square_bbox

CO3D_DIR = "data/co3d_v1"
CO3D_ANNOTATION_DIR = "data/co3dv1_annotations"

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Co3dv1Dataset(Dataset):
    def __init__(
        self,
        category=("all",),
        split="train",
        transform=None,
        debug=False,
        random_aug=True,
        jitter_scale=(1.1, 1.2),
        jitter_trans=(-0.07, 0.07),
        num_images=2,
    ):
        """
        Args:
            num_images: Number of images in each batch.
            perspective_correction (str):
                "none": No perspective correction.
                "warp": Warp the image and label.
                "label_only": Correct the label only.
        """
        if "all" in category:
            category = TRAINING_CATEGORIES
        category = sorted(category)

        if split == "train":
            split_name = "train_known"
        elif split == "test":
            split_name = "test_known"

        self.rotations = {}
        self.category_map = {}
        for c in category:
            annotation_file = osp.join(CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
            for seq_name, seq_data in annotation.items():
                if len(seq_data) < 2:
                    continue
                filtered_data = []
                self.category_map[seq_name] = c
                for data in seq_data:
                    # Ignore all unnecessary information.
                    filtered_data.append(
                        {
                            "filepath": data["filepath"],
                            "bbox": data["bbox"],
                            "R": data["R"],
                            "focal_length": data["focal_length"],
                        },
                    )
                self.rotations[seq_name] = filtered_data

        self.sequence_list = list(self.rotations.keys())
        self.split = split
        self.debug = debug
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform
        if random_aug:
            self.jitter_scale = jitter_scale
            self.jitter_trans = jitter_trans
        else:
            self.jitter_scale = [1.15, 1.15]
            self.jitter_trans = [0, 0]
        self.num_images = num_images

    def __len__(self):
        return len(self.sequence_list)

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def __getitem__(self, index):
        sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]
        ids = np.random.choice(len(metadata), self.num_images)
        if self.debug:
            # id1, id2 = np.random.choice(5, 2, replace=False)
            pass
        return self.get_data(index=index, ids=ids)

    def get_data(self, index=None, sequence_name=None, ids=(0, 1)):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        metadata = self.rotations[sequence_name]

        annos = [metadata[i] for i in ids]
        images = [Image.open(osp.join(CO3D_DIR, anno["filepath"])) for anno in annos]
        rotations = [torch.tensor(anno["R"]) for anno in annos]

        additional_data = {}

        images_transformed = []
        for anno, image in zip(annos, images):
            if self.transform is None:
                images_transformed.append(image)
            else:
                bbox = np.array(anno["bbox"])
                bbox_jitter = self._jitter_bbox(bbox)
                image = self._crop_image(image, bbox_jitter)
                images_transformed.append(self.transform(image))
        images = images_transformed

        relative_rotation = rotations[0].T @ rotations[1]
        category = self.category_map[sequence_name]
        batch = {
            "relative_rotation": relative_rotation,
            "model_id": sequence_name,
            "category": category,
            "n": len(metadata),
        }
        if self.transform is None:
            batch["image"] = images
        else:
            batch["image"] = torch.stack(images)
        batch["ind"] = torch.tensor(ids)
        batch["R"] = torch.stack(rotations)
        batch.update(additional_data)
        return batch
