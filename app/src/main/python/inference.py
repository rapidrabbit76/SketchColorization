import random
from glob import glob
from os import makedev, path
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageOps
import onnxruntime
from transforms import Lambda, Compose, ToTensor, Resize, Normalize
from image4layer import Image4Layer
import torch
from torch.nn.functional import interpolate


class InferenceHandler:
    """ TorchServe Handler for PaintsTorch"""

    def __init__(self, content=None):

        get = content.get_resource

        self.__model = onnxruntime.InferenceSession(
            get(content.model_path))

        self.line_transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize([0.5], [0.5]),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.hint_transform = Compose([
            # input must RGBA !
            Resize((128, 128), Image.NEAREST),
            Lambda(lambda img: img.convert(mode='RGB')),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.line_draft_transform = Compose([
            Resize((128, 128)),
            ToTensor(),
            Normalize([0.5], [0.5]),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.alpha_transform = Compose([
            Lambda(lambda img: self.get_alpha(img)),
        ])

    def convert_to_pil_image(self, image):
        image = np.transpose(image, (0, 2, 3, 1))
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.astype(np.uint8)[0]
        image = Image.fromarray(image).convert('RGB')
        return image

    def convert_to_pil_line(self, image, size=512):
        image = np.transpose(image, (0, 2, 3, 1))
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.astype(np.uint8)[0]
        image = np.reshape(image, (size, size))
        image = Image.fromarray(image).convert('RGB')
        return image

    def get_alpha(self, hint: Image.Image):
        """
        :param hint:
        :return:
        """
        hint = hint.resize((128, 128), Image.NEAREST)
        hint = np.array(hint)
        alpha = hint[:, :, -1]
        alpha = np.expand_dims(alpha, 0)
        alpha = np.expand_dims(alpha, 0).astype(np.float32)
        alpha[alpha > 0] = 1.0
        alpha[alpha > 0] = 1.0
        alpha[alpha < 1.0] = 0
        return alpha

    def prepare(self, line: Image.Image, hint: Image.Image):
        """
        :param req:
        :return:
        """

        line = line.convert(mode='L')
        alpha = hint.convert(mode='RGBA')
        hint = hint.convert(mode='RGBA')

        w, h = line.size

        alpha = self.alpha_transform(alpha)
        line_draft = self.line_draft_transform(line)
        line = self.line_transform(line)
        hint = self.hint_transform(hint)
        hint = hint * alpha
        hint = np.concatenate([hint, alpha], 1)
        return line, line_draft, hint, (w, h)

    def inference(self, data, **kwargs):
        """
        PaintsTorch inference
        colorization Line Art Image
        :param data: tuple (line, line_draft, hint, size)
        :return: tuple image, size(w,h)
        """
        line, line_draft, hint = data

        inputs_tag = self.__model.get_inputs()
        inputs = {
            inputs_tag[0].name: line,
            inputs_tag[1].name: line_draft,
            inputs_tag[2].name: hint
        }
        image = self.__model.run(None, inputs)[0]
        return image

    def resize(self, image: Image.Image, size: tuple) -> Image.Image:
        """
        Image resize to 512
        :param image: PIL Image data
        :param size:  w,h tuple
        :return: resized Image
        """
        (width, height) = size

        if width > height:
            rate = width / height
            new_height = 512
            new_width = int(512 * rate)
        else:
            rate = height / width
            new_width = 512
            new_height = int(rate * 512)

        return image.resize((new_width, new_height), Image.BICUBIC)

    def postprocess(self, data) -> Image.Image:
        """
        POST processing from inference image Tensor
        :param data: tuple image, size(w,h)
        :return: processed Image json
        """
        pred, size = data
        pred = self.convert_to_pil_image(pred)
        image = self.resize(pred, size)
        return image


__HANDLER__: InferenceHandler = None


def predict(line: Image.Image, hint: Image.Image, connect_info: tuple):
    line, line_draft, hint, size = __HANDLER__.prepare(line, hint)
    pred = __HANDLER__.inference((line, line_draft, hint))
    image = __HANDLER__.postprocess((pred, size))
    return image
