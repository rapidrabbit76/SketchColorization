import typing as T
from service_streamer import ThreadedStreamer

from core.settings import get_settings
from fastapi import Depends
from loguru import logger
from PIL import Image, ImageOps
import numpy as np


from .manager import build_streamer, Task
from .transforms import Lambda, Compose, ToTensor, Resize, Normalize


env = get_settings()


class SketchColorizationService:
    def __init__(
        self,
        streamer: ThreadedStreamer = Depends(build_streamer),
    ) -> None:
        logger.info(f"DI:{self.__class__.__name__}")
        self.streamer = streamer

        self.line_transform = Compose(
            [
                Resize((512, 512)),
                ToTensor(),
                Normalize([0.5], [0.5]),
                Lambda(lambda img: np.expand_dims(img, 0)),
            ]
        )
        self.hint_transform = Compose(
            [
                # input must RGBA !
                Resize((128, 128), Image.NEAREST),
                Lambda(lambda img: img.convert(mode="RGB")),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                Lambda(lambda img: np.expand_dims(img, 0)),
            ]
        )
        self.line_draft_transform = Compose(
            [
                Resize((128, 128)),
                ToTensor(),
                Normalize([0.5], [0.5]),
                Lambda(lambda img: np.expand_dims(img, 0)),
            ]
        )
        self.alpha_transform = Compose(
            [
                Lambda(lambda img: self.get_alpha_cahannel(img)),
            ]
        )

    def predict(self, line: Image.Image, hint: Image.Image) -> Image.Image:

        origin_size = line.size
        line = line.convert(mode="L")
        alpha = hint.convert(mode="RGBA")
        hint = hint.convert(mode="RGBA")

        if line.size != hint.size:
            hint = hint.resize(line.size, Image.NEAREST)

        alpha = self.alpha_transform(alpha)
        line_draft = self.line_draft_transform(line)
        line = self.line_transform(line)
        hint = self.hint_transform(hint)
        hint = hint * alpha
        hint = np.concatenate([hint, alpha], 1)

        colored = self.streamer.predict(
            [Task(line=line, line_draft=line_draft, hint=hint)]
        )[0]
        colored = self.convert_to_pil_image(colored)
        colored = self.resize(colored, origin_size)
        return colored

    @staticmethod
    def get_alpha_cahannel(hint: Image.Image) -> np.ndarray:
        hint = hint.resize((128, 128), Image.NEAREST)
        hint = np.array(hint)
        alpha = hint[:, :, -1]
        alpha = np.expand_dims(alpha, 0)
        alpha = np.expand_dims(alpha, 0).astype(np.float32)
        alpha[alpha > 0] = 1.0
        alpha[alpha > 0] = 1.0
        alpha[alpha < 1.0] = 0
        return alpha

    @staticmethod
    def convert_to_pil_image(image: np.ndarray) -> Image.Image:
        image = np.transpose(image, (0, 2, 3, 1))
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.astype(np.uint8)[0]
        image = Image.fromarray(image).convert("RGB")
        return image

    @staticmethod
    def resize(image: Image.Image, size: tuple) -> Image.Image:
        (w, h) = size

        if w > h:
            rate = w / h
            width, height = int(512 * rate), 512
            return image.resize((width, height), Image.BICUBIC)

        rate = h / w
        width, height = 512, int(rate * 512)
        return image.resize((width, height), Image.BICUBIC)
