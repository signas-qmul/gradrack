import torch

from gradrack.oscillators import Oscillator


class Sinusoid(Oscillator):
    def _generate(self, phase):
        return torch.sin(phase)
