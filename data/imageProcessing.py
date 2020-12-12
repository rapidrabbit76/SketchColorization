import random
from typing import Tuple
import colorsys
import math
import numbers
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Resize, CenterCrop, ToPILImage, ToTensor, Normalize


class DraftArgumentation:
    """ Draft Model Output Image Argumentation """

    def __init__(self, device):
        self.denomal = Denormalize()
        self.to_tensor = ToTensor()
        self.norm = Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))
        self.spay = RandomSpray()
        self.t2i = ToPILImage()
        self.device = device

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.6:
            images = self.denomal(images)
            images = images.cpu()
            new_images = []
            for image in images:
                image = self.t2i(image)
                image = self.spay(image)
                image = self.to_tensor(image)
                image = self.norm(image)
                image = torch.unsqueeze(image, 0)
                new_images.append(image)
            images = torch.cat(new_images, 0).to(self.device)
        return images


class RandomSpray:
    """ Color Spay Image Argumentation"""

    def __call__(self, image: Image.Image) -> Image.Image:
        ori_img = image
        color = self.get_dominant_color(ori_img)
        # Random Color Spray
        img = np.array(ori_img)

        h = int(img.shape[0] / 30)
        w = int(img.shape[1] / 30)
        a_x = np.random.randint(0, h)
        a_y = np.random.randint(0, w)
        b_x = np.random.randint(0, h)
        b_y = np.random.randint(0, w)
        begin_point = np.array([min(a_x, b_x), a_y])
        end_point = np.array([max(a_x, b_x), b_y])
        tan = (begin_point[1] - end_point[1]) / \
            (begin_point[0] - end_point[0] + 0.001)

        center_point_list = []
        for i in range(begin_point[0], end_point[0] + 1):
            a = i
            b = (i - begin_point[0]) * tan + begin_point[1]
            center_point_list.append(np.array([int(a), int(b)]))
        center_point_list = np.array(center_point_list)

        lamda = random.uniform(0.01, 10)
        paper = np.zeros((h, w, 3))
        mask = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                dis = self.min_dis([i, j], center_point_list)
                paper[i, j, :] = np.array(color) / np.exp(lamda * dis)
                mask[i, j] = np.array([255]) / np.exp(lamda * dis)

        paper = (paper).astype('uint8')
        mask = (mask).astype('uint8')

        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        im = cv2.resize(
            paper, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        imq = Image.fromarray(im)
        imp = ori_img.copy()
        imp.paste(
            imq, (0, 0, imp.size[0], imp.size[1]), mask=Image.fromarray(mask))
        return imp

    def min_dis(self, point, point_list: list):
        dis = []
        for p in point_list:
            dis.append(
                np.sqrt(np.sum(np.square(np.array(point) - np.array(p)))))
        return min(dis)

    def get_dominant_color(self, image: Image.Image):
        image = image.convert('RGBA')
        image.thumbnail((200, 200))
        max_score = 0
        dominant_color = 0

        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):

            if a == 0:
                continue

            saturation = colorsys.rgb_to_hsv(
                r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b *
                        802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            if y > 0.9:
                continue

            if ((r > 230) & (g > 230) & (b > 230)):
                continue

            score = (saturation + 0.1) * count

            if score > max_score:
                max_score = score
                dominant_color = (r, g, b)

        return dominant_color


class RandomCrop:
    """ Image Pairs Randomly Crop
    """

    def __init__(self, size: int):
        """ 
        Args:
            size (int): Crop Size
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1: Image, img2: Image):
        w, h = img1.size
        th, tw = self.size

        if w == tw and h == th:
            return img1, img2

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))
        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class Tensor2Image():
    def __init__(self):
        self.__t2p = ToPILImage()

    def __call__(self, images):
        new_images = []
        for image in images:
            img = self.__t2p(image).convert("RGB")
            new_images.append(img)
        return new_images


class Denormalize:

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        :param tesnsor: tensor range -1 to 1
        :return: tensor range 0 to 1
        '''
        tensor = tensor.cpu()
        return (tensor + 1.0) / 2.0


class Kmeans:
    def __init__(self, k: int):
        self.__k = k

    def __call__(self, image: Image):
        image = np.array(image)
        z = image.reshape(-1, 3)
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(
            z, self.__k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        image = res.reshape(image.shape)
        return Image.fromarray(image)


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


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


def random_flip(color: Image,
                line: Image) -> (Image, Image):
    if random.random() > 0.5:
        color = color.transpose(Image.FLIP_LEFT_RIGHT)
        line = line.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.5:
        color = color.transpose(Image.FLIP_TOP_BOTTOM)
        line = line.transpose(Image.FLIP_TOP_BOTTOM)

    return (color, line)
