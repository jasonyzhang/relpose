"""
Dataset class for custom datasets. Should provide a directory with images and
optionally a directory of masks. The masks are used to extracting bounding boxes for
each image. If masks are not provided, bounding boxes must be provided directly instead.

Directory format:

image_dir
|_ image0001.jpg
mask_dir
|_ mask0001.png
"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from relpose.utils.bbox import mask_to_bbox, square_bbox


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, bboxes=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bboxes = []
        self.images = []

        for image_path in sorted(os.listdir(image_dir)):
            self.images.append(Image.open(osp.join(image_dir, image_path)))
        self.n = len(self.images)
        if bboxes is None:
            for mask_path in sorted(os.listdir(mask_dir))[: self.n]:
                mask = plt.imread(osp.join(mask_dir, mask_path))
                if len(mask.shape) == 3:
                    mask = mask[:, :, :3]
                else:
                    mask = np.dstack([mask, mask, mask])
                self.bboxes.append(mask_to_bbox(mask))
        else:
            self.bboxes = bboxes
        self.jitter_scale = [1.15, 1.15]
        self.jitter_trans = [0, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return 1

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
        # Should use get_data instead.
        ids = np.random.choice(self.n, 2)
        return self.get_data(ids=ids)

    def get_data(self, ids=(0, 1)):
        images = [self.images[i] for i in ids]
        bboxes = [self.bboxes[i] for i in ids]
        images_transformed = []
        for _, (bbox, image) in enumerate(zip(bboxes, images)):
            bbox = np.array(bbox)
            bbox_jitter = self._jitter_bbox(bbox)
            image = self._crop_image(image, bbox_jitter)
            images_transformed.append(self.transform(image))
        images = images_transformed
        return torch.stack(images)
