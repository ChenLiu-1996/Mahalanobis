from typing import Tuple
import random
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter


class SingleInstanceTwoView:
    '''
    This class is adapted from BarlowTwins and SimSiam.
    '''

    def __init__(self, imsize: int, mean: Tuple[float], std: Tuple[float]):
        self.augmentation = transforms.Compose([
            transforms.Resize(
                imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ],
                                   p=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        aug1 = self.augmentation(x)
        aug2 = self.augmentation(x)
        return aug1, aug2


class GaussianBlur(object):

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
