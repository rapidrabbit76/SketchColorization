import sys
import typing as T

from pydantic import BaseSettings


class ModelSetting(BaseSettings):
    MODEL_PATH: str = "model_store/SketchColorizationModel.onnx"


class MicroBatchSettings(BaseSettings):
    MB_BATCH_SIZE = 1
    MB_MAX_LATENCY = 0.01
    MB_TIMEOUT = 120


class Settings(
    ModelSetting,
    MicroBatchSettings,
):
    CORS_ALLOW_ORIGINS: T.List[str] = ["*"]
