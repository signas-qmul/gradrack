import math

import torch


class ADSR(torch.nn.Module):
    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        self._validate_input(attack, decay, release)
        time = torch.ones_like(gate).cumsum(dim=-1) - 1.0

        # calculate attack portion of envelope
        attack_slope = time / attack
        attack_mask = (time < attack) * gate
        attack_section = attack_slope * attack_mask

        # calculate decay portion of envelope
        decay_time_constant = 1 / (math.e ** (1 / decay))

        decay_slope = decay_time_constant ** (time - attack)
        decay_mask = (time >= attack) * gate
        decay_section = decay_slope * decay_mask

        return attack_section + decay_section

    def _validate_input(self, attack, decay, release):
        if attack < 0:
            raise ValueError('Attack length must be >= 0')
        if decay == 0:
            raise ValueError('Decay length must be > 0')
        if release == 0:
            raise ValueError('Release length must be > 0')
