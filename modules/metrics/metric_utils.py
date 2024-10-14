import numpy as np
from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class RunningMean():
    """Computes and stores the average """

    initialized:bool = field(init=False, default=False)
    avg:float        = field(init=False, default=None)
    sum:float        = field(init=False, default=None)
    count:float      = field(init=False, default=None)

    def initialize(self, val, weight=1):
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val:float, weight:float=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val:float, weight:float=1):
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def average(self) -> float:
        return self.avg

    def reset(self):
        self.initialized = False
        self.avg = None
        self.sum = None
        self.count = None

@dataclass
class Mean():
    """Computes and stores the average """

    initialized:bool = field(init=False, default=False)
    avg:float        = field(init=False, default=None)
    sum:float        = field(init=False, default=None)
    vals:List[float] = field(init=False, default=None)

    def initialize(self, val):
        self.vals = [val]
        self.avg = val
        self.sum = val
        self.initialized = True

    def update(self, val:float):
        if not self.initialized:
            self.initialize(val)
        else:
            self.add(val)

    def add(self, val:float):
        self.sum += val
        self.vals += [val]
        self.avg = self.sum / len(self.vals)

    def average(self) -> float:
        return self.avg

    def reset(self):
        self.initialized = False
        self.avg = None
        self.sum = None
        self.vals = None


