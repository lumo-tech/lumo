from abc import ABC
from typing import Dict, Any, Optional


class RecordBackend(ABC):
    """
    Defines an abstract base class for recording and logging data.

    This module provides an abstract base class, RecordBackend,
    that defines the interface for recording and logging data.

    Subclasses of RecordBackend must implement the log(), log_image(), log_audio(), and log_video() methods.
    """

    def __init__(self, location, *args, **kwargs):
        self.location = location

    def log(self, data: Dict[str, Any],
            step: Optional[int] = None,
            commit: Optional[bool] = None,
            sync: Optional[bool] = None):
        """
        Logs data to the backend.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to be logged.
            step (Optional[int]): The step number associated with the data.
            commit (Optional[bool]): Whether to commit the data to storage.
            sync (Optional[bool]): Whether to synchronize the data across multiple devices.
        """
        raise NotImplementedError()

    def log_image(self, image_array, caption=None):
        """
        Logs an image to the backend.

        Args:
            image_array (ndarray): A NumPy array representing the image to be logged.
            caption (Optional[str]): A caption describing the image.
        """
        raise NotImplementedError()

    def log_audio(self, image_array, caption=None):
        """
        Logs an audio clip to the backend.

        Args:
            audio_array (ndarray): A NumPy array representing the audio clip to be logged.
            caption (Optional[str]): A caption describing the audio clip.
        """
        raise NotImplementedError()

    def log_image(self, image_array, caption=None):
        """
        Logs a video clip to the backend.

        Args:
            video_array (ndarray): A NumPy array representing the video clip to be logged.
            caption (Optional[str]): A caption describing the video clip.
        """
        raise NotImplementedError()
