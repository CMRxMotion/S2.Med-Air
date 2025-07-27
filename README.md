## Introduction
This is the official code submitted to the CMRxMotion challenge by Team Med-Air. 
It's also the Final Dockerfile submitted to MICCAI 2022 [CMRxMotion Challenge](http://cmr.miccai.cloud/)

<!-- arxiv Version with the GitHub code link in the paper:
[]()

Springer Version without GitHub code link in the paper:
[]() -->


## Datasets
CMRxMotion Dataset: [http://cmr.miccai.cloud/data/](http://cmr.miccai.cloud/data/)

## Methodology and Poster Overview
<!-- ![Poster](poster.png) -->
<img src="./poster.png" alt="Poster" width="1600">

## Usage
This repository has been made publicly available with the consent of Team Med-Air under the Apache 2.0 License.

## Citation
If this code is useful for your research, please consider citing:

```
@InProceedings{10.1007/978-3-031-23443-9_47,
author="Gong, Shizhan
and Lu, Weitao
and Xie, Jize
and Zhang, Xiaofan
and Zhang, Shaoting
and Dou, Qi",
editor="Camara, Oscar
and Puyol-Ant{\'o}n, Esther
and Qin, Chen
and Sermesant, Maxime
and Suinesiaputra, Avan
and Wang, Shuo
and Young, Alistair",
title="Robust Cardiac MRI Segmentation with Data-Centric Models to Improve Performance via Intensive Pre-training and Augmentation",
booktitle="Statistical Atlases and Computational Models of the Heart. Regular and CMRxMotion Challenge Papers",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="494--504",
abstract="Segmentation of anatomical structures from Cardiac Magnetic Resonance (CMR) is central to the non-invasive quantitative assessment of cardiac function and structure, and deep-learning-based automatic segmentation models prove to have satisfying performance. However, patients' respiratory motion during the scanning process can greatly degenerate the quality of CMR images, resulting in a serious performance drop for deep learning algorithms. Building a robust cardiac MRI segmentation model is one of the keys to facilitating the use of deep learning in practical clinic scenarios. To this end, we experiment with several network architectures and compare their segmentation accuracy and robustness to respiratory motion. We further pre-train our network on large publicly available CMR datasets and augment our training set with adversarial augmentation, both methods bring significant improvement. We evaluate our methods on the cine MRI dataset of the CMRxMotion challenge and obtain promising performance for the segmentation of the left ventricle, left ventricular myocardium, and right ventricle.",
isbn="978-3-031-23443-9"
}
```
