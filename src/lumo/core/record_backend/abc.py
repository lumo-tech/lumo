from abc import ABC
from typing import Dict, Any, Optional


class RecordBackend(ABC):
    def __init__(self, location, *args, **kwargs):
        self.location = location

    def log(self, data: Dict[str, Any],
            step: Optional[int] = None,
            commit: Optional[bool] = None,
            sync: Optional[bool] = None):
        raise NotImplementedError()

    def log_image(self, image_array, caption=None):
        raise NotImplementedError()
    def log_audio(self, image_array, caption=None):
        raise NotImplementedError()
    def log_image(self, image_array, caption=None):
        raise NotImplementedError()
