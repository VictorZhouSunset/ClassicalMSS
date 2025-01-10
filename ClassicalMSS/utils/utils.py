# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs

from concurrent.futures import CancelledError
from contextlib import contextmanager
import math
import os
import tempfile
import typing as tp

import torch
from torch.nn import functional as F
from torch.utils.data import Subset

def pull_metric(history: tp.List[dict], name: str):
    out = []
    for metrics in history:
        metric = metrics
        for part in name.split("."): # Each time it loops, it goes deeper one layer into the dictionary
            metric = metric[part]
        out.append(metric)
    return out


class DummyPoolExecutor:
    """
    A mock implemetation of python's concurrent.futures.ProcessPoolExecutor which
    is used to run tasks in parallel;
    This mock implementation is used when the same API is used but no parallelization is needed.
    That's the reason for the name DummyPoolExecutor and so many dummy variables in the class
    """
    class DummyResult:
        def __init__(self, func, _dict, *args, **kwargs):
            self.func = func
            self._dict = _dict
            self.args = args
            self.kwargs = kwargs

        def result(self):
            if self._dict["run"]:
                return self.func(*self.args, **self.kwargs)
            else:
                raise CancelledError()

    def __init__(self, workers=0):
        self._dict = {"run": True}

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, self._dict, *args, **kwargs)

    def shutdown(self, *_, **__):
        self._dict["run"] = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return
