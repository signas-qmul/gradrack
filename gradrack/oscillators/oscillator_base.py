from abc import ABC, abstractmethod

import torch


class Oscillator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, frequency, phase_mod, length=None, sample_rate=None):
        pass

    @abstractmethod
    def other_method(self):
        pass
