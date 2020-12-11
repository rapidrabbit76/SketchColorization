import os
import random
from glob import glob
from typing import Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda, Normalize, Resize, ToTensor, Compose, ColorJitter, \
    RandomChoice
from torchvision.transforms.functional import to_tensor
from transforms import RandomCrop
from abc import ABCMeta
from data import xdog, dilate_abs_line


class ImageLoaderBase(ABCMeta):
    """ Image Data Loader Base Class """

    @staticmethod
    def _jitter(line: torch.Tensor) -> torch.Tensor:
        """ Line Arts Image Luminance Data-Augmentation

        Args:
            line (Tensor): line-arts image Tensor

        Returns:
            Tensor: line-arts data 
        """
        ran = random.uniform(0.7, 1)
        line = line * ran + (1 - ran)
        return torch.clamp(
            line, min=0, max=1)

    @staticmethod
    def _create_mask(zero_p: float = 0.4) -> torch.Tensor:
        """ Create Random Hint Mask (0 or 1 binary 2D Mask)

        Args:
            zero_p (float): 
            Probability flag for zero hint count  Defaults to 0.4.

        Returns:
            Tensor: Binary 2D Hint Mask 
        """
        hint_count = 0

        if random.random() < zero_p:
            if random.random() < 0.4:
                hint_count = random.randint(1, 5)
        else:
            hint_count = random.randint(
                random.randint(5, 32),
                random.randint(42, 65))

        area = 128 * 128

        zero = np.zeros(shape=[area - hint_count], dtype=np.uint8)
        one = np.ones(shape=[hint_count], dtype=np.uint8)
        mask = np.concatenate([zero, one], -1)
        np.random.shuffle(mask)
        mask = np.reshape(mask, newshape=[128, 128]) * 255
        _, mask = cv2.threshold(mask,
                                127, 255,
                                cv2.THRESH_BINARY)
        return to_tensor(mask)

    @staticmethod
    def _create_line(image: Image) -> Image:
        """[summary]

        Args:
            image (Image): Color PIL Image (target Image)

        Returns:
            Image: Greyscale PIL Image (Line-Arts)
        """
        if random.random() > 0.5:
            return xdog(image)
        else:
            return dilate_abs_line(image)
