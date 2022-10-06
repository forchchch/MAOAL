import os
import random
import torch

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from PIL import Image
from torchvision.datasets import CIFAR100

class CIFAR100MTL(CIFAR100):
    """
    CIFAR100 dataset, 20 tasks (grouped by coarse labels)
    Each task is a 5-label classification, with 2500 training and 500 testing number of data for each task.
    Modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(self, root, subset_id=0, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100MTL, self).__init__(root, train, transform, target_transform, download)
        # define coarse label maps
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

        self.coarse_targets = coarse_labels[self.targets]

        # filter the data and targets for the desired subset
        self.data = self.data[self.coarse_targets == subset_id]
        self.targets = np.array(self.targets)[self.coarse_targets == subset_id]

        # remap fine labels into 5-class classification
        self.targets = np.unique(self.targets, return_inverse=True)[1]

        # update semantic classes
        self.class_dict = {
            "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
            "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
            "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
            "food containers": ["bottle", "bowl", "can", "cup", "plate"],
            "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
            "household electrical device": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
            "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
            "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
            "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
            "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
            "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
            "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
            "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
            "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
            "people": ["baby", "boy", "girl", "man", "woman"],
            "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
            "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
            "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
            "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
            "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
        }

        self.subset_class = list(self.class_dict.keys())[subset_id]
        self.classes = self.class_dict[self.subset_class]