# Dataset Preparation

## Preparing CO3Dv1 Dataset

Follow the directions to download and extract CO3Dv1 from
[here](https://github.com/facebookresearch/co3d/tree/v1).

You will need to pre-process that dataset to extract the bounding boxes and camera poses.
The bounding boxes are processed separately because it takes the most amount of time
and should only be run once.
```
python -m preprocess.preprocess_co3dv1 --category all --precompute_bbox \
    --co3d_v1_dir /path/to/co3d_v1
python -m preprocess.preprocess_co3dv1 --category all \
    --co3d_v1_dir /path/to/co3d_v1
```

## Preparing Your Own Dataset

For inference on your own video, you can use the `CustomDataset` class in
`relpose/dataset/custom.py`. You will need to provide a directory of images and a
directory of masks or bounding boxes. The masks are simply used to compute bounding
boxes.
