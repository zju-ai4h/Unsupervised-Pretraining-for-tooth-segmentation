# Unsupervised-Pre-training
Unsupervised Pre-training Improves Tooth Segmentation in 3-Dimensional Intraoral Mesh Scans (Accept at MIDL2022)
# Abstract
Accurate tooth segmentation in 3-Dimensional (3D) intraoral scanned (IOS) mesh data
is an essential step for many practical dental applications. Recent research highlights the
success of deep learning based methods for end-to-end 3D tooth segmentation, yet most
of them are only trained or validated with a small dataset as annotating 3D IOS dental
surfaces requires complex pipelines and intensive human efforts. In this paper, we propose
a novel method to boost the performance of 3D tooth segmentation leveraging large-scale
unlabeled IOS data. Our tooth segmentation network is first pre-trained with an unsupervised learning framework and point-wise contrastive learning loss on the large-scale
unlabeled dataset and subsequently fine-tuned on a small labeled dataset. With the same
amount of annotated samples, our method can achieve a mIoU of 89.38%, significantly
outperforming the supervised counterpart. Moreover, our method can achieve better performance with only 40% of the annotated samples as compared to the fully supervised
baselines. To the best of our knowledge, we present the first attempt of unsupervised pre-training for 3D tooth segmentation, demonstrating its strong potential in reducing human
efforts for annotation and verification.
# Getting Started
## Environment
* Ubuntu 18.04
* CUDA 11.2
* Pytorch 1.9.0
* Open3d 0.9.0
* Trimesh 3.9.35
* MinkowskiEngine 0.5.4
## Train on your dataset
### Pretrain
For pre-training, put your dataset list to ``pretrain/dataset`` and rename it as ``overlap.txt``.
#### Dataset format
dataset

├── 00000.stl  
 ……  
└── XXXXX.stl
The ``.stl`` is a simple 3-Dimensional Intraoral Mesh Scan without label.
### Finetune
For fine-tuning, our codebase is based
