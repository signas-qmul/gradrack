import math

import torch
import torch.nn.functional as F


class ADSR(torch.nn.Module):
    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        self._validate_input(attack, decay, release)

        # compute time axes
        attack_decay_axis = self._cumsum_resetting_on_zeros(gate)
        release_axis = self._cumsum_resetting_on_zeros(1 - gate)

        # compute envelope segment masks
        attack_mask = (attack_decay_axis <= attack) * gate
        decay_mask = (attack_decay_axis > attack) * gate
        release_mask = 1 - gate

        # compute exponential time constants
        decay_time_constant = 1 / (math.e**(1 / decay))
        release_time_constant = 1 / (math.e**(1 / release))

        # calculate slope functions
        decay_slope = decay_time_constant**(attack_decay_axis - attack)
        decay_slope = sustain + (1 - sustain) * decay_slope

        release_slope = self._calculate_release_slope(release_time_constant,
                                                      decay_slope, gate,
                                                      decay_mask, release_mask)

        attack_start = release_slope * attack_mask.roll(-1, -1) * release_mask
        attack_start = F.pad(attack_start, (0, 1))
        attack_start = attack_start.roll(1, -1)
        attack_start = attack_start.narrow(-1, 0, attack_start.shape[-1] - 1)
        attack_start = attack_start.cumsum(-1)
        attack_slope = ((1 - attack_start) * attack_decay_axis / attack +
                        attack_start)

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

    def _cumsum_resetting_on_zeros(self, gate):
        accumulated_gate = gate.cumsum(dim=-1)
        accum_gate_lo = accumulated_gate[gate == 0]
        padded_accum_gate_lo = F.pad(accum_gate_lo, (1, 0))
        gate_lo_diff = padded_accum_gate_lo[1:] - padded_accum_gate_lo[:-1]
        pre_axis_gate = gate.clone()
        pre_axis_gate[gate == 0] = -gate_lo_diff
        time_axis = pre_axis_gate.cumsum(dim=-1)

        return time_axis

    def _shift_tensor_along_dim(self, tensor, dim, shift_amount=1):
        x = tensor.transpose(0, dim)
        x = F.pad(x, (shift_amount, 0))
        x = x[:-shift_amount]
        x = x.transpose(dim, 0)

        return x

    def _calculate_release_slope(self, release_time_constant, decay_slope,
                                 gate, decay_mask, release_mask):

        # find final value of each preceding decay section
        release_initial = decay_slope * release_mask.roll(-1, -1) * decay_mask

        # move that final value into the first position of the decay slope
        release_initial = self._shift_tensor_along_dim(release_initial, -1)

        # populate the remaining values of the decay slope with ones
        release_initial[release_initial == 0] = 1
        # remove any ones that precede the first attack
        release_initial = release_initial * (gate.cumsum(-1) > 0)
        # apply the release mask and apply the time constant non-cumulatively
        release_initial = (release_initial * release_mask *
                           release_time_constant)
        # switch to a log scale
        log_release_start = torch.log(release_initial)
        # replace infs with zeros to allow cumsum algorithm to work
        log_release_start[log_release_start == float('-Inf')] = 0
        # use cumsum with reset on zeros alogrithm to accumulate log release
        cum_log_release_start = self._cumsum_resetting_on_zeros(
            log_release_start)
        # convert zeros back to negative infs
        cum_log_release_start[cum_log_release_start == 0] = float('-Inf')
        # convert back to linear domain (cumsum becomes cumprod)
        release_slope = torch.exp(cum_log_release_start)

        return release_slope
