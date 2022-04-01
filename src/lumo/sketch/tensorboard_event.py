from tensorboard.backend.event_processing import event_accumulator
from typing import List, Optional, Dict
from collections import defaultdict


class TBReader:
    def __init__(self, root: str, name: Optional[str] = None):
        self.event = event_accumulator.EventAccumulator(root,
                                                        size_guidance={  # see below regarding this argument
                                                            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                            event_accumulator.IMAGES: 4,
                                                            event_accumulator.AUDIO: 4,
                                                            event_accumulator.SCALARS: 0,
                                                            event_accumulator.HISTOGRAMS: 1,
                                                        })

        self.root = root
        self.name = name

    def __enter__(self):
        self.event.Reload()

    def scalars(self) -> List[str]:
        with self:
            return self.event.Tags()['scalars']


class GroupTBReader:
    def __init__(self, roots: List[str], names: Optional[List[str]] = None):
        if names is None:
            names = [f'Tz{i}' for i in range(len(roots))]
        assert len(roots) == len(names)
        self.readers = [TBReader(i) for i in roots]
        self.names = names

    def scalars(self) -> Dict[str, List[str]]:
        res = defaultdict(list)
        for r, n in zip(self.readers, self.names):
            for tag in r.scalars():
                res[tag].append(n)
        return res

    def get_table_of(self, scalar_tag):
        """"""
        pass
