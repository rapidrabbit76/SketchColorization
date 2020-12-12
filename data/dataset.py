import random
from typing import Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import Lambda, Normalize, Resize, ToTensor, Compose, ColorJitter
from torchvision.transforms.functional import to_tensor
from abc import ABCMeta
from data import xdog, dilate_abs_line, RandomCrop, random_flip


class DatasetBase:
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
        """ Create Line-arts Image

        Args:
            image (Image): Color PIL Image (target Image)

        Returns:
            Image: Greyscale PIL Image (Line-Arts)
        """
        if random.random() > 0.5:
            return xdog(image).convert('L')
        else:
            return dilate_abs_line(image).convert('L')


class DraftModelDataset(Dataset, DatasetBase):
    def __init__(self,
                 image_paths: list,
                 training: bool,
                 size: int = 128):

        self._image_paths = image_paths
        self._training = training
        self._random_crop = RandomCrop(512)
        self._color_compose = Compose([
            Resize(size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5))
        ])

        compose_processing = [
            Resize(size),
            ToTensor()
        ]

        if training:
            compose_processing.append(Lambda(self._jitter))

        compose_processing.append(Normalize([0.5], [0.5]))

        self._line_compose = Compose(compose_processing)

        self._color_jitter = ColorJitter(brightness=0,
                                         contrast=0.1,
                                         saturation=0.1,
                                         hue=0.03)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, item) -> (torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor):

        target_image = Image.open(self._image_paths[item]).convert('RGB')
        if random.random() > 0.0001:
            line_image = self._create_line(target_image)
        else:
            # Data argumentation color image to greyscale image
            line_image = target_image.convert('L')

        target_image, line_image = self._random_crop(target_image, line_image)

        # Data argumentation
        if self._training is True:
            target_image, line_image = self._argumentation(
                target_image, line_image)
            mask = self._create_mask()
        else:
            mask = self._create_mask(0)

        # Preprocessing
        target_image = self._color_compose(target_image)
        line_image = self._line_compose(line_image)

        # Build Hint
        hint = target_image.clone()
        hint = hint * mask
        hint_image = torch.cat([hint, mask], 0)

        return target_image, hint_image, line_image

    def _argumentation(self,
                       target: Image,
                       line: Image) -> (Image, Image):
        """ Data Argumentataion """
        line = ImageOps.equalize(line) if random.random() >= 0.5 else line

        target, line = random_flip(target, line)
        target = self._color_jitter(target)
        return target, line


class ColorizationModelDataset(Dataset, DatasetBase):
    def __init__(self, image_paths: list, training: bool):
        self.image_paths = image_paths

        self._hint_compos = Compose([
            Resize(128),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5))
        ])

        self._color_compose = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5))
        ])
        self._line_compose = Compose([
            ToTensor(),
            Lambda(self._jitter),
            Normalize([0.5], [0.5]),
        ])

        self._line_draft_compose = Compose([
            Resize(128),
            ToTensor(),
            Lambda(self._jitter),
            Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor):

        target_image = Image.open(self._image_paths[item]).convert('RGB')
        line_image = self._create_line(target_image)

        target_image, line_image = self._random_crop(target_image,
                                                     line_image)

        if self._training is True:
            target_image, line_image = self._argumentation(target_image,
                                                           line_image)

        hint_image = self._hint_compos(target_image)
        target_image = self._color_compose(target_image)
        line_draft = self._line_draft_compose(line_image)
        line_image = self._line_compose(line_image)

        mask = self._create_mask()
        hint_image = torch.cat([hint_image * mask, mask], 0)

        return target_image, hint_image, line_image, line_draft

    @ staticmethod
    def _create_mask() -> torch.Tensor:
        area = 128 * 128
        hint_count = 128

        zero = np.zeros(shape=[area - hint_count], dtype=np.uint8)
        one = np.ones(shape=[hint_count], dtype=np.uint8)
        mask = np.concatenate([zero, one], -1)
        np.random.shuffle(mask)
        mask = np.reshape(mask, newshape=[128, 128]) * 255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return to_tensor(mask)


class AutoEncoderDataset(Dataset, DatasetBase):
    def __init__(self, image_paths: list, training: bool):
        """
        :param image_paths:color_image_path list
        :param training:   boolean flag for DataArgumentation
        :param size:       resize size
        """
        self._image_paths = image_paths
        self._random_crop = RandomCrop(512)
        self._image_compose = Compose([
            Resize(128),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5),
                      (0.5, 0.5, 0.5))
        ])
        self._line_compose = Compose([
            Resize(128),
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

    def __getitem__(self, item) -> (Image, Image):
        target = Image.open(self._image_paths[item]).convert("RGB")
        line = self._create_line(target)
        target, line = self._random_crop(target, line)
        target, line = self._argumentation(target, line)
        target = self._image_compose(target)
        line = self._line_compose(line)
        return target, line

    def __len__(self):
        return len(self._image_paths)

    def _argumentation(self,
                       target: Image,
                       line: Image) -> (Image, Image):
        """ Data Argumentataion """
        target, line = random_flip(target, line)
        target = self._color_jitter(target)
        return target, line
