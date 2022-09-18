from functools import lru_cache
import typing as T
import onnxruntime
from loguru import logger
import numpy as np
from pydantic import BaseModel
from service_streamer import ThreadedStreamer
from core.settings import get_settings


class Task(BaseModel):
    line: np.ndarray
    line_draft: np.ndarray
    hint: np.ndarray

    class Config:
        arbitrary_types_allowed = True


env = get_settings()


class SketchColorizationModel:
    def __init__(self) -> None:
        self._model = onnxruntime.InferenceSession(env.MODEL_PATH)
        self.inputs_tag = self._model.get_inputs()
        logger.info("model loadded")

    def predict(self, batch: T.List[Task]) -> np.ndarray:
        logger.info("inference start")
        logger.info(f"batch_size: {len(batch)}")
        line = []
        line_draft = []
        hint = []

        for task in batch:
            line.append(task.line)
            line_draft.append(task.line_draft)
            hint.append(task.hint)

        line, line_draft, hint = map(
            lambda x: np.concatenate(x, 0), [line, line_draft, hint]
        )

        inputs = {
            self.inputs_tag[0].name: line,
            self.inputs_tag[1].name: line_draft,
            self.inputs_tag[2].name: hint,
        }
        images = self._model.run(None, inputs)
        logger.info("inference end")
        return images


@lru_cache(maxsize=1)
def build_streamer():
    model = SketchColorizationModel()
    env = get_settings()
    streamer = ThreadedStreamer(
        model.predict,
        batch_size=env.MB_BATCH_SIZE,
        max_latency=env.MB_MAX_LATENCY,
    )
    logger.info("streamer init")
    return streamer
