# RelPose: Predicting Probabilistic Relative Rotation for Single Objects in the Wild

[[`arXiv`](https://arxiv.org/abs/2208.05963)]
[[`Project Page`](https://jasonyzhang.com/relpose/)]
[[`Bibtex`](#citing-relpose)]

## Installation

Follow directions for setting up CO3D (v1 or v2) from [here](dataset.md)

### Setup
We recommend using conda to manage dependencies. Make sure to install a cudatoolkit
compatible with your GPU.
```
git clone --depth 1 https://github.com/jasonyzhang/relpose.git
conda create -n relpose python=3.8
conda activate relpose
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Model Weights

You can download the pre-trained model weights on both CO3Dv1 and CO3Dv2 from
[Google Drive](https://drive.google.com/file/d/1XwRjxOzqj6DXGg_bzYFy83iDlZx8mkQ-/view?usp=share_link).
Alternatively, you can use gdown:
```
gdown --output data/pretrained_relpose.zip https://drive.google.com/uc?id=1XwRjxOzqj6DXGg_bzYFy83iDlZx8mkQ-
unzip data/pretrained_relpose.zip -d data
```

### Installing Pytorch3d

Here, we list the recommended steps for installing Pytorch3d. Refer to the 
[official installation directions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
for troubleshooting and additional details.

```
mkdir -p external
git clone --depth 1 --branch v0.7.0 https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
cd external/pytorch3d
conda activate relpose
conda install -c conda-forge -c fvcore -c iopath -c bottler fvcore iopath nvidiacub
python setup.py install
```

If you need to compile for multiple architectures (e.g. Turing for 2080TI and Maxwell
for 1080TI), you can pass the architectures as an environment variable, i.e. 
`TORCH_CUDA_ARCH_LIST="Maxwell;Pascal;Turing;Volta" python setup.py install`.

If you get a warning about the default C/C++ compiler on your machine, you should
compile Pytorch3D using the same compiler that your pytorch installation uses, likely
gcc/g++. Try: `CC=gcc CXX=g++ python setup.py install`.


### Dataset Preparation

Please see [docs/dataset.md](docs/dataset.md) for instructions on preparing the CO3Dv1 dataset or your own dataset.

## Training

Once the datasets are setup, run the following command to train on 4 GPUs on CO3Dv2:
```
python -m relpose.trainer --batch_size 64 --num_gpus 4 --output_dir output --dataset co3d
```

With 4 2080TI GPUs, we expect training to take a little less than 2 days.

## Inference

Please see [notebooks/demo.ipynb](notebooks/demo.ipynb) for a demo of visualizing
pairwise relative pose distributions given 2 images as well as recovering camera
rotations using the pairwise predictor. Currently, the demo supports using a Maximum
Spanning Tree and Coordinate Ascent for joint camera pose inference.

## Evaluation

Please see [docs/eval.md](docs/eval.md) for instructions on evaluating on sequential,
MST, and coordinate ascent inference.

## <a name="CitingRelPose"></a>Citing RelPose

If you use find this code helpful, please cite:

```BibTeX
@InProceedings{zhang2022relpose,
    title = {{RelPose}: Predicting Probabilistic Relative Rotation for Single Objects in the Wild},
    author = {Zhang, Jason Y. and Ramanan, Deva and Tulsiani, Shubham},
    booktitle = {European Conference on Computer Vision},
    year = {2022},
}
```