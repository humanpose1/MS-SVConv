# MS-SVConv : 3D Point Cloud Registration with Multi-Scale Architecture and Self-supervised Fine-tuning

Compute features for 3D point cloud registration. The article is available on [Arxiv](https://arxiv.org/abs/2103.14533) .
It relies on:
- A multi scale sparse voxel architecture
- Self-supervised fine-tuning
The combination of both allows better generalization capabilities and transfer across different datasets.

The code is available on the [torch-points3d repository](https://github.com/nicolas-chaulet/torch-points3d).
This repository is to show how to launch the code for training and testing.

## Demo
If you want to try MS-SVConv without installing anything on your computer, A Google colab notebook is available here (it takes few minutes to install everything). In the colab, we compute features using MS-SVConv and use Ransac (implementation of Open3D) to compute the transformation.
You can try on 3DMatch on ETH.

## Installation

The code have been tried on an NVDIA RTX 1080 Ti with CUDA version 10.1. The OS was Ubuntu 18.04.

### Quick installation

This installation is for those who just want to quickly use MS-SVConv on there project, if you want to train MS-SVConv or evaluate it, please, visit the next section.
First create a virtual environnement, and install all these packages (it is like google colab):
```

```

### Installation for training and evaluation
This installation step is necessary if you want to train and evaluate MS-SVConv.


first you need, to clone the [torch-points3d repository](https://github.com/nicolas-chaulet/torch-points3d)
```
git clone https://github.com/nicolas-chaulet/torch-points3d.git
```
Torch-points3d uses [poetry](https://python-poetry.org/) to manage the packages. after installing Poetry, run :
```
poetry install --no-root
```
Activate the environnement
```
poetry shell
```
If you want to train MS-SVConv on 3DMatch, you will need pycuda (It's optional for testing).
```
pip install pycuda
```
You will also need to install [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) and [torchsparse](https://github.com/mit-han-lab/torchsparse)
Finally, you will need [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) for testing.

If you have problems with installation (espaecially with [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)), please visit the [Troubleshooting section](https://github.com/nicolas-chaulet/torch-points3d#troubleshooting) of torch-points3d page.

## Training

### registration


If you want to train MS-SVConv with 3 heads starting at the scale 2cm, run this command:
```
poetry run python train.py
```

automatically, the code will call the right yaml file with the right task.
If you just want to train MS-SVConv with 1 head, run this command
```
```
You can modify some hyperparameters directly on the command line. For example, if you want to change the learning rate, you can run:
```

```

To resume training:


WARNING : On 3DMatch, you will need a lot of disk space because the code will download the RGBD image on 3DMatch and build the fragments from scratch. Also the code takes time (few hours).
To train on Modelnet run this command:
```
poetry run
```

For 3DMatch, it was supervised training because the pose is necessary. But we can also fine-tune in a self-supervised fashion (without needing the pose).
To fine-tune on ETH run this command:
```
```
To fine-tune on TUM, run this command:
```
```
You can also fine-tune on 3DMatch, run this command:
```
```

For all these command, it will save in `outputs` directory log of the training, it will save a `.pt` file which is the weights of

### semantic segmentation

You can also train MS-SVConv on scannet for semantic segmentation. To do this simply run:
```
```
And you can transfer from registration to segmantation, with this command:
```
```

## Evaluation

If you want to evaluate the models on 3DMatch, run:

```
poetry run
```
on ETH,
```
poetry run
```
on TUM:
```
poetry run
```
You can also visualize matches, you can run:
```

```

You should obtain this image

## Model Zoo
You can find all the pretrained model  (More will be added in the future)

## citation

If you like our work, please cite it :
```
@misc{horache2021mssvconv,
      title={3D Point Cloud Registration with Multi-Scale Architecture and Self-supervised Fine-tuning},
      author={Sofiane Horache and Jean-Emmanuel Deschaud and François Goulette},
      year={2021},
      eprint={2103.14533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## TODO
 - Add other pretrained models
 - Add others datasets such as KITTI
 - add tutorial for a new dataset
