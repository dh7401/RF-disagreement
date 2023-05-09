# [ICML 2023] Demystifying Disagreement-on-the-Line in High Dimensions
arXiv link: https://arxiv.org/abs/2301.13371

`experiments/simulations`: Section 3, 4\
`experiments/real_datasets`: Section 5


For the real-world dataset experiments, the datasets should be downloaded from:\
`CIFAR-10-C`: https://zenodo.org/record/2535967#.Y9hoeOzMI1M \
`Tiny ImageNet`: https://www.kaggle.com/c/tiny-imagenet \
`Tiny ImageNet-C`: https://zenodo.org/record/2469796#.Y9hoei-B1QI \
`CIFAR-10` and `Camelyon17` are downloaded from the packages `torchvision` and `wilds` (https://github.com/p-lambda/wilds)

Downloaded datasets should be placed as
```
experiments
---real_datasets
------data
---------CIFAR10
---------CIFAR10-C
---------TinyImageNet
---------TinyImageNet-C
---------camelyon17_v1.0
```
