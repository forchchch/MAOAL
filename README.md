# Module-Aware Optimization for Auxiliary Learning
by Hong Chen, Xin Wang, Yue Liu, Chaoyu Guan and Wenwu Zhu.

## Introduction
This work can be used when you want to optimize a model for the target task with the help of several auxiliary losses. It can automatically find how much each auxiliary loss should contribute to each part of the whole model, preventing the potential module-wise conflicts. Some of the auxilearn codes are modified from the [AuxLearn Repo](https://github.com/AvivNavon/AuxiLearn), we thank them for offering the code. 
![_](./MAOAL.png)
## Dataset
The provided code is for reproducing the CIFAR-100/20 experiment in the paper. Create a folder named data in the repo using:
```
mkdir data
```
Later the used dataset will be downloaded to this folder.

## Run the experiment
You can run the script to reproduce the CIFAR-100/20 experiments. The argument "exp_name" is the name for the experiment. 
```
sh train.sh
```
Additionally, if you want to run some of the baselines or tune hyperparameters, change the params.json in the config folder. Note that the default config requires 15GB GPU memory, you can choose smaller batchsize to avoid 'out of memory' error.
### Explanation for key configs
    + ['main']['lr']: learning rate for the lower optimization
    + ['hyper']['lr]: learning rate for the upper optimization
    + use_aux: 1 for methods using bi-level optimization, 0 for methods that do not use bi-level optimization
    + interval: iteration between two upper optimization
    + mode: 'modular(MAOAL)', 'GCS', 'common'(SLL/Equal, change ['main']['aux_weight] to 0.0 for SLL, 1.0 for Equal), 'aux'(AuxL)

## Lower and Upper Optimization
    + lower: The importance parameterized gradient is implemented in the hypermodel class.
    + upper: In line 257-line 288, the upper optimization is conducted. gauxlearn package includes the algorithm for upper gradient calculation.


You may also find the papers in the citation useful.

## Citation
```
@inproceedings{chen2022auxiliary,
title = {Auxiliary Learning with Joint Task and Data Scheduling},
author = {Chen, Hong and Wang Xin, and Guan, Chaoyu and Liu, Yue and Zhu Wenwu},
booktitle = {International Conference on Machine Learning},
pages = {3634--3647},
year = {2022},
organization = {PMLR}
}
```




