import math

import torch
import torch.nn.functional as F


class ADSR(torch.nn.Module):
    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        self._validate_input(attack, decay, release)

        # compute time axes
        attack_decay_axis = self._compute_time_axes(gate)
        release_axis = self._compute_time_axes(1 - gate)

        # compute envelope segment masks
        attack_mask = (attack_decay_axis <= attack) * gate
        decay_mask = (attack_decay_axis > attack) * gate
        release_mask = 1 - gate

        # compute exponential time constants
        decay_time_constant = 1 / (math.e ** (1 / decay))
        release_time_constant = 1 / (math.e ** (1 / release))

        # calculate slope functions
        attack_slope = attack_decay_axis / attack

        decay_slope = decay_time_constant ** (attack_decay_axis - attack)
        decay_slope = sustain + (1 - sustain) * decay_slope

        release_start = decay_slope * release_mask.roll(-1, -1) * decay_mask
        release_start = release_start.roll(1, -1)
        release_start = release_start.cumsum(-1)
        release_start = (sustain
                         + decay_time_constant
                         * (release_start - sustain))
        release_slope = release_start * release_time_constant ** (release_axis)

        # mask slopes
        attack_section = attack_slope * attack_mask
        decay_section = decay_slope * decay_mask
        release_section = release_slope * release_mask

        return attack_section + decay_section + release_section

    def _validate_input(self, attack, decay, release):
        if attack == 0:
            raise ValueError('Attack length must be > 0')
        if decay == 0:
            raise ValueError('Decay length must be > 0')
        if release == 0:
            raise ValueError('Release length must be > 0')

    def _compute_time_axes(self, gate):
        accumulated_gate = gate.cumsum(dim=-1)
        accum_gate_lo = accumulated_gate[gate == 0]
        padded_accum_gate_lo = F.pad(accum_gate_lo, (1, 0))
        gate_lo_diff = padded_accum_gate_lo[1:] - padded_accum_gate_lo[:-1]
        pre_axis_gate = gate.clone()
        pre_axis_gate[gate == 0] = -gate_lo_diff
        time_axis = pre_axis_gate.cumsum(dim=-1)

        return time_axis
