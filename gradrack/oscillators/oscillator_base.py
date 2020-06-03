from abc import ABC, abstractmethod
import math

import torch


class Oscillator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, frequency, phase_mod, length=None, sample_rate=None):
        instantaneous_frequency = frequency.repeat_interleave(length, dim=-1)
        phase = instantaneous_frequency.cumsum(-1)
        phase = phase - phase.select(-1, 0).unsqueeze(-1)
        phase = math.tau * phase / sample_rate

        phase = phase + phase_mod

        self.generate(phase)

    @abstractmethod
    def generate(self, phase):
        pass
