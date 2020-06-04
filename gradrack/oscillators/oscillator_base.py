from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F


class Oscillator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, frequency, phase_mod, length=None, sample_rate=None):
        if length is None:
            if phase_mod.shape[-1] > 1:
                length = phase_mod.shape[-1]
            elif frequency.shape[-1] == 1 and phase_mod.shape[-1] == 1:
                raise LengthMismatchError("Sample length must be provided " +
                                          "for scalar frequency and " +
                                          "phase_mod parameters")
        else:
            if frequency.shape[-1] > 1 or phase_mod.shape[-1] > 1:
                raise LengthMismatchError("Can't use length parameter when " +
                                          "a time dimension is provided for " +
                                          "frequency or phase.")

        if frequency.shape[-1] == 1:
            frequency =\
                frequency.repeat_interleave(length, dim=-1)

        original_frequency_length = frequency.shape[-1]
        frequency = F.pad(frequency, (1, 0), 'constant', 0)
        frequency = frequency.narrow(-1, 0, original_frequency_length)

        phase = frequency.cumsum(-1)
        phase = math.tau * phase / sample_rate

        phase = phase + phase_mod

        self.generate(phase)

    @abstractmethod
    def generate(self, phase):
        pass


class LengthMismatchError(Exception):
    pass
