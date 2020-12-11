import random
import cv2
import numpy as np
from PIL import Image


def dilate_abs_line(image: Image) -> Image:
    image = np.asarray(image)
    k = 5 if random.random() >= 0.5 else 4
    kernel = np.ones([k, k], dtype=np.uint8)
    dilated = cv2.dilate(image, kernel=kernel)
    diff = cv2.absdiff(dilated, image)
    line = 255 - diff
    return Image.fromarray(line)


def xdog(img: Image, k_sigma: float = 4.5,
         p: float = 0.95, epsilon: float = -0.1,
         phi: float = 200) -> Image:

    img = np.asarray(img)
    sigma = random.choice([0.3, 0.4, 0.5])

    def soft_threshold(si, epsilon, phi):
        t = np.zeros(si.shape)
        si_bright = si >= epsilon
        si_dark = si < epsilon
        t[si_dark] = 1.0
        t[si_bright] = 1.0 + np.tanh(phi * (si[si_bright]))
        return t

    def _xdog(img_1, sigma, k_sigma, p, epsilon, phi):
        s = dog(img_1, sigma, k_sigma, p)
        t = soft_threshold(s, epsilon, phi)
        return (t * 127.5).astype(np.uint8)

    def dog(img_2, sigma, k_sigma, p):
        sigma_large = sigma * k_sigma
        g_small = cv2.GaussianBlur(img_2, (0, 0), sigma)
        g_large = cv2.GaussianBlur(img_2, (0, 0), sigma_large)
        s = g_small - p * g_large
        return s

    line = _xdog(img, sigma=sigma, k_sigma=k_sigma,
                 p=p, epsilon=epsilon, phi=phi)
    return Image.fromarray(line)


def random_flip(color: Image, line: Image) -> (Image, Image):
    if random.random() > 0.5:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)
        line = line.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.5:
        color = color.transpose(Image.FLIP_TOP_BOTTOM)
        line = line.transpose(Image.FLIP_TOP_BOTTOM)

    return (color, line)
