from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F


class Oscillator(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, frequency, phase_mod, length=None, sample_rate=None):
        self.check_input_shape(frequency, phase_mod, length)
        frequency = self.broadcast_dimensions(frequency, phase_mod, length)
        phase = self.compute_phase(frequency, phase_mod, sample_rate)

        self.generate(phase)

    def check_input_shape(self, frequency, phase_mod, length):
        if length is None:
            if frequency.shape[-1] == 1 and phase_mod.shape[-1] == 1:
                raise LengthMismatchError("Sample length must be provided " +
                                          "for scalar frequency and " +
                                          "phase_mod parameters")
        elif frequency.shape[-1] > 1 or phase_mod.shape[-1] > 1:
            raise LengthMismatchError("Can't use length parameter when " +
                                      "a time dimension is provided for " +
                                      "frequency or phase.")

    def broadcast_dimensions(self, frequency, phase_mod, length):
        if length is None and phase_mod.shape[-1] > 1:
            length = phase_mod.shape[-1]

        if frequency.shape[-1] == 1:
            frequency =\
                frequency.repeat_interleave(length, dim=-1)

        original_frequency_length = frequency.shape[-1]
        frequency = F.pad(frequency, (1, 0), 'constant', 0)
        frequency = frequency.narrow(-1, 0, original_frequency_length)

        return frequency

    def compute_phase(self, frequency, phase_mod, sample_rate):
        phase = frequency.cumsum(-1)
        phase = math.tau * phase / sample_rate
        phase = phase + phase_mod
        return phase


    @abstractmethod
    def generate(self, phase):
        pass


class LengthMismatchError(Exception):
    pass
