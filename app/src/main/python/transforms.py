import numpy as np
from PIL import Image


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, image: Image.Image) -> np.ndarray:
        image = np.array(image, dtype=np.float32)
        image /= 255.0
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        image = np.transpose(image, [2, 0, 1])
        return image


class Resize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, self.interpolation)


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float)

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        mean = self.mean
        std = self.std

        if mean.ndim == 1:
            mean = mean[:, None, None]
        if std.ndim == 1:
            std = std[:, None, None]
        tensor -= mean
        tensor /= std
        return tensor
