"""
for reading values recorded by tensorboard directly instead of opening tensorboard web pages.
"""
import pprint as pp
from collections import namedtuple
from typing import List


Scalars = namedtuple('ScalarEvent', ['wall_times', 'values', 'steps'])


class BoardReader():
    """
    ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])
    CompressedHistogramEvent = namedtuple('CompressedHistogramEvent',
                                          ['wall_time', 'step',
                                           'compressed_histogram_values'])
    HistogramEvent = namedtuple('HistogramEvent',
                                ['wall_time', 'step', 'histogram_value'])
    HistogramValue = namedtuple('HistogramValue', ['min', 'max', 'num', 'sum',
                                                   'sum_squares', 'bucket_limit',
                                                   'bucket'])
    ImageEvent = namedtuple('ImageEvent', ['wall_time', 'step',
                                           'encoded_image_string', 'width',
                                           'height'])
    AudioEvent = namedtuple('AudioEvent', ['wall_time', 'step',
                                           'encoded_audio_string', 'content_type',
                                           'sample_rate', 'length_frames'])
    TensorEvent = namedtuple('TensorEvent', ['wall_time', 'step', 'tensor_proto'])
    """

    def __init__(self, file):
        from tensorboard.backend.event_processing import event_accumulator
        self.ea = event_accumulator.EventAccumulator(file,
                                                     size_guidance={  # see below regarding this argument
                                                         event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                         event_accumulator.IMAGES: 4,
                                                         event_accumulator.AUDIO: 4,
                                                         event_accumulator.SCALARS: 0,
                                                         event_accumulator.HISTOGRAMS: 1,
                                                     })
        self._reloaded = False

    def _check_reload(self):
        """load values from dist to memory"""
        if not self._reloaded:
            self.ea.Reload()

    def get_scalars(self, tag) -> Scalars:
        """get scalars named '<tag>' in this board"""
        self._check_reload()
        wall_times, steps, values = list(zip(*self.ea.Scalars(tag)))  # 'wall_time', 'step', 'value'
        values = [float("{:.4f}".format(i)) for i in values]
        return Scalars(wall_times, values, steps)

    @property
    def scalars_tags(self) -> List[str]:
        """get all scalar tags"""
        self._check_reload()
        return self.tags['scalars']

    @property
    def tags(self) -> dict:
        """get all tags in this board"""
        self._check_reload()
        return self.ea.Tags()

    def summary(self):
        """print tags in this board"""
        self._check_reload()
        pp.pprint(self.ea.Tags())
