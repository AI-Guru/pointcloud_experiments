# Pointcloud Experiments with ModelNet40. Includes PointNet and GAPNet

The purpose of this repository is to provide a clean implementation of GAPNet. In order to facilitate comparability, it includes PointNet as a baseline. The goal is to come up with a readable implementation of GAPNet that outperforms PointNet.

# Always star if you like.

If you enjoy this repo, please give it a star. That would be very appreciated!

# Getting in touch.

If you got any bug reports or feature requests, please open an issue here on GitHub.

- [Subscribe to my YouTube channel](https://www.youtube.com/channel/UCcMEBxcDM034JyJ8J3cggRg?view_as=subscriber).
- [Become a patron](https://www.patreon.com/ai_guru).
- [Subscribe to my newsletter](http://ai-guru.de/newsletter/).
- [Visit by homepage/blog](http://ai-guru.de/).
- [Join me on Slack](https://join.slack.com/t/ai-guru/shared_invite/enQtNDEzNjUwMTIwODM0LTdlOWQ1ZTUyZmQ5YTczOTUxYzk2YWI4ZmE0NTdmZGQxMmUxYmUwYmRhMDg1ZDU0NTUxMDI2OWVkOGFjYTViOGQ).
- [Add me on LinkedIn](https://www.linkedin.com/in/dr-tristan-behrens-ai-guru-734967a2/).
- [Add me on Facebook](https://www.facebook.com/AIGuruTristanBehrens).


## Acknowledgements.

Based heavily on:

- [https://github.com/TianzhongSong/PointNet-Keras](https://github.com/TianzhongSong/PointNet-Keras)
- [https://github.com/FrankCAN/GAPointNet](https://github.com/FrankCAN/GAPointNet)

Those people did some really, really great work!

## Networks.

### 1. PointNet.

Reference: [https://arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)

PointNet is included as a baseline.

### 2. GAPNet.

Reference: [https://arxiv.org/abs/1905.08705](https://arxiv.org/abs/1905.08705)

GAPNet combines the Graph Neural Network approach with attention.

![GAPNet](https://raw.githubusercontent.com/AI-Guru/pointcloud_experiments/master/resources/GAPNet%20network.png)

## How to run.

Download ModelNet40 data:

`sh download_modelnet40.sh`

Prepare data:

`python prepare_data.py`

Train PointNet:

`python train_cls.py pointnet`

Train GAPNet draft:

`python train_cls.py gapnet_dev`

## Notes.

This project is under heavy construction.
