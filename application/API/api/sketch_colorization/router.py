import typing as T

from fastapi import Form, Depends, UploadFile, File, Response
from fastapi_restful.cbv import cbv
from fastapi_restful.inferring_router import InferringRouter
from PIL import Image

from .request import read_image, image2bytes
from .response import StableDiffussionResponse
from app.sketch_colorization.service import SketchColorizationService
from core.settings import get_settings


router = InferringRouter()
env = get_settings()


@cbv(router)
class SketchColorization:
    svc: SketchColorizationService = Depends()

    @router.post("/paint", response_model=StableDiffussionResponse)
    def text2image(
        self,
        line: UploadFile = File(...),
        hint: UploadFile = File(...),
    ):
        line = read_image(line)
        hint = read_image(hint)
        colored = self.svc.predict(line=line, hint=hint)
        image_bytes = image2bytes(image=colored)
        return Response(image_bytes, media_type="image/webp")
