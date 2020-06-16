import math

import torch
import torch.nn.functional as F


class ADSR(torch.nn.Module):
    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        self._validate_input(attack, decay, release)

        # compute time axes
        attack_decay_axis = self._cumsum_resetting_on_value(gate)

        # compute envelope segment masks
        attack_mask = (attack_decay_axis <= attack) * gate
        decay_mask = (attack_decay_axis > attack) * gate
        release_mask = 1 - gate

        # compute exponential time constants
        decay_time_constant = 1 / (math.e**(1 / decay))
        release_time_constant = 1 / (math.e**(1 / release))

        # calculate slope functions
        attack_slope = attack_decay_axis / attack

        decay_slope = decay_time_constant**(attack_decay_axis - attack)
        decay_slope = sustain + (1 - sustain) * decay_slope

        for _ in range(2):
            ad_slope = attack_mask * attack_slope + decay_mask * decay_slope

            release_slope = self._calculate_release_slope(
                release_time_constant, ad_slope, gate,
                attack_mask + decay_mask, release_mask)
            attack_slope = self._calculate_attack_slope(
                attack, attack_mask, release_mask, release_slope,
                attack_decay_axis)

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

    def _cumsum_resetting_on_value(self, t, reset_value=0):
        accumulated = t.clone()
        accumulated[t == reset_value] = 0
        accumulated = accumulated.cumsum(dim=-1)
        accumulated_lo = accumulated[t == reset_value]
        padded_accum_lo = F.pad(accumulated_lo, (1, 0))
        padded_accum_lo_diff = padded_accum_lo[1:] - padded_accum_lo[:-1]
        output = t.clone()
        output[t == reset_value] = -padded_accum_lo_diff
        output = output.cumsum(dim=-1)
        output[t == reset_value] = reset_value

        return output

    def _shift_tensor_along_dim(self, tensor, dim, shift_amount=1):
        x = tensor.transpose(0, dim)
        x = F.pad(x, (shift_amount, 0))
        x = x[:-shift_amount]
        x = x.transpose(dim, 0)

        return x

    def _calculate_release_slope(self, release_time_constant, ad_slope,
                                 gate, decay_mask, release_mask):
        # find final value of each preceding decay section
        release_initial = ad_slope * release_mask.roll(-1, -1) * decay_mask

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
        cum_log_release_start = self._cumsum_resetting_on_value(
            log_release_start)
        # convert zeros back to negative infs
        cum_log_release_start[cum_log_release_start == 0] = float('-Inf')
        # convert back to linear domain (cumsum becomes cumprod)
        release_slope = torch.exp(cum_log_release_start)

        return release_slope

    def _calculate_attack_slope(self, attack, attack_mask, release_mask,
                                release_slope, attack_decay_axis):
        # find final value of preceding release section
        attack_start = release_slope * attack_mask.roll(-1, -1) * release_mask
        attack_start = F.pad(attack_start, (0, 1))
        # invert values so we are describing range of attack
        attack_range = 1 - attack_start
        # move values into first attack position
        attack_range = attack_range.roll(1, -1)
        attack_range = attack_range.narrow(-1, 0, attack_range.shape[-1] - 1)
        # remove anything outside of attack segments
        attack_range = attack_range * attack_mask

        # take the log and cumulatively add, resetting each time we reach a
        # -inf [i.e. log(0)]. This has the effect of 'spreading' each attack
        # start value over its corresponding attack segment
        log_attack_range = torch.log(attack_range)
        cum_log_attack_range = self._cumsum_resetting_on_value(
            log_attack_range, float('-Inf'))
        # convert back to linear scale (additions become multiplications)
        attack_range = torch.exp(cum_log_attack_range)

        # use our new range axis to compute the attack slope
        attack_slope = attack_range * attack_decay_axis / attack + (
            1 - attack_range)

        return attack_slope
