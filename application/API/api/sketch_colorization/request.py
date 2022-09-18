from fastapi import UploadFile, HTTPException, status
from PIL import Image
import io
from loguru import logger


def read_image(image: UploadFile) -> Image.Image:
    try:
        image = Image.open(image.file)
    except Exception as e:
        logger.error(f"{image.filename}: ,{e}")
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"""{image.filename} is not image file  """,
        )
    return image


def image2bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="WEBP")
    return buf.getvalue()
