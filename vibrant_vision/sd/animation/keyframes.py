from collections.abc import MutableSequence

import numpy as np
import pandas as pd


class KeyFrames(MutableSequence):
    def __init__(self, total_frames, any_type=False) -> None:
        super(KeyFrames, self).__init__()
        if any_type:
            self._schedule_series = pd.Series([None for _ in range(total_frames)])
        else:
            self._schedule_series = pd.Series([np.nan for _ in range(total_frames)])

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._schedule_series}>"

    def __len__(self):
        return len(self._schedule_series)

    def __getitem__(self, index):
        return self._schedule_series[index]

    def __delitem__(self, index):
        self._schedule_series[index] = np.nan

    def __setitem__(self, index, value):
        self._schedule_series[index] = value

    def __str__(self) -> str:
        return self._schedule_series.to_string()

    def insert(self, index, value):
        self._schedule_series[index] = value

    def interpolate(self, method="quadratic"):
        self._schedule_series.interpolate(inplace=True, method=method, limit_direction="both")
