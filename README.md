# Pointcloud Experiments with ModelNet40. Includes PointNet and a draft of GAPNet

## Acknowledgements.

Based heavily on:

- [https://github.com/TianzhongSong/PointNet-Keras](https://github.com/TianzhongSong/PointNet-Keras)
- [https://github.com/FrankCAN/GAPointNet](https://github.com/FrankCAN/GAPointNet)

Those people did some really, really great work!

## References.

- PointNet: [https://arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)
- GAPNet: [https://arxiv.org/abs/1905.08705](https://arxiv.org/abs/1905.08705)

## Intentions.

The purpose of this repository is to provide a clean implementation of GAPNet. In order to allow comparability, it includes PointNet. The goal is to come up with a readable implementation of GAPNet that outperforms PointNet.

## How to run.

Download ModelNet40 data:

`sh download_modelnet40.sh`

Prepare data:

`python prepare_data.py`

Train PointNet:

`python train_cls pointnet`

Train GAPNet draft:

`python train_cls gapnet_dev`

## Notes.

This project is under heavy construction.