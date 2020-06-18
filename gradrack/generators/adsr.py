import math

import torch
import torch.nn.functional as F


class ADSR(torch.nn.Module):
    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        if sample_rate is not None:
            attack *= sample_rate
            decay *= sample_rate
            release *= sample_rate

        attack, decay, sustain, release = self._ensure_tensors(
            attack, decay, sustain, release
        )
        self._validate_input(attack, decay, release)

        # compute time axis
        attack_decay_axis = self._cumsum_resetting_on_value(gate)

        # compute envelope segment masks
        attack_mask = (attack_decay_axis <= attack) * gate
        decay_mask = (attack_decay_axis > attack) * gate
        release_mask = 1 - gate

        # compute exponential time constants
        decay_time_constant = 1 / (math.e ** (1 / decay))
        release_time_constant = 1 / (math.e ** (1 / release))

        # calculate slope functions
        # (start with attack regions counting from zero)
        attack_slope = attack_decay_axis / attack

        decay_slope = decay_time_constant ** (attack_decay_axis - attack)
        decay_slope = sustain + (1 - sustain) * decay_slope

        # Alternately update attack and release segments:
        for _ in range(2):
            ad_slope = attack_mask * attack_slope + decay_mask * decay_slope

            release_slope = self._calculate_release_slope(
                release_time_constant,
                ad_slope,
                gate,
                attack_mask + decay_mask,
                release_mask,
            )
            attack_slope = self._calculate_attack_slope(
                attack,
                attack_mask,
                release_mask,
                release_slope,
                attack_decay_axis,
            )

        # mask slopes
        attack_section = attack_slope * attack_mask
        decay_section = decay_slope * decay_mask
        release_section = release_slope * release_mask

        envelope = attack_section + decay_section + release_section
        return envelope

    def _ensure_tensors(self, attack, decay, sustain, release):
        return (
            self._ensure_tensor(attack),
            self._ensure_tensor(decay),
            self._ensure_tensor(sustain),
            self._ensure_tensor(release),
        )

    def _ensure_tensor(self, input):
        return (
            torch.Tensor([input])
            if not isinstance(input, torch.Tensor)
            else input
        )

    def _validate_input(self, attack, decay, release):
        """Ensure no erroneous values are input"""
        if (attack == 0).any():
            raise ValueError("Attack length must be > 0")
        if (decay == 0).any():
            raise ValueError("Decay length must be > 0")
        if (release == 0).any():
            raise ValueError("Release length must be > 0")

    def _cumsum_resetting_on_value(self, t, reset_value=0):
        """A moderately hacky algorithm for performing a cumsum whose count
        resets every time a zero is reached"""
        accumulated = t.clone()
        accumulated[t == reset_value] = 0
        accumulated = accumulated.cumsum(dim=-1)

        accumulated_lo = accumulated[t == reset_value]
        padded_accum_lo = F.pad(accumulated_lo, (1, 0))
        padded_accum_lo = padded_accum_lo.transpose(0, -1)
        padded_accum_lo_diff = padded_accum_lo[1:] - padded_accum_lo[:-1]
        padded_accum_lo_diff = padded_accum_lo_diff.transpose(0, -1)

        output = t.clone()
        output[t == reset_value] = -padded_accum_lo_diff

        output = output.transpose(0, -1)
        output[0] = accumulated.transpose(0, -1)[0]
        output = output.transpose(0, -1)

        output = output.cumsum(dim=-1)
        output[t == reset_value] = reset_value

        return output

    def _cumprod_resetting_on_zero(self, t):
        """Independently calculates the cumulative product of each region of
        the tensor separated by zeros. Calculated across the last dim."""
        log_t = torch.log(t)
        cum_log_t = self._cumsum_resetting_on_value(log_t, float("-Inf"))
        return torch.exp(cum_log_t)

    def _shift_tensor_along_dim_with_zero_padding(
        self, tensor, dim, shift_amount=1
    ):
        """Shift a tensor along a dimension and zero pad the remainder."""
        x = tensor.transpose(-1, dim)
        x = F.pad(x, (shift_amount, 0))
        x = x.transpose(dim, -1)
        x = x.narrow(dim, 0, x.shape[dim] - shift_amount)

        return x

    def _find_starting_values(self, previous_slope, previous_mask, this_mask):
        """Find the starting values of all regions covered by a particular
        mask by overlapping it with the preceding section's mask."""
        starting_values = (
            previous_slope * this_mask.roll(-1, -1) * previous_mask
        )
        starting_values = self._shift_tensor_along_dim_with_zero_padding(
            starting_values, -1
        )
        return starting_values

    def _calculate_release_slope(
        self, release_time_constant, ad_slope, gate, ad_mask, release_mask
    ):
        """Compute the release slopes for all areas covered by the release
        mask, taking the ending value of each previous attack/decay slope as
        the starting value."""
        release_initial = self._find_starting_values(
            ad_slope, ad_mask, release_mask
        )

        # populate the remaining values of the release slope with ones
        release_initial[release_initial == 0] = 1
        release_initial = release_initial * release_mask

        # remove any ones that precede the first attack
        release_initial = release_initial * (gate.cumsum(-1) > 0)
        # apply the time constant non-cumulatively
        release_initial = release_initial * release_time_constant

        # calculate each independent release slope
        release_slope = self._cumprod_resetting_on_zero(release_initial)

        return release_slope

    def _calculate_attack_slope(
        self,
        attack,
        attack_mask,
        release_mask,
        release_slope,
        attack_decay_axis,
    ):
        """Compute the attack slopes for all areas covered by the attack mask,
        taking the ending value of each previous release slope as the
        starting value."""
        attack_initial = self._find_starting_values(
            release_slope, release_mask, attack_mask
        )

        # invert values so we are describing range of attack
        attack_range = 1 - attack_initial
        # remove anything outside of attack segments
        attack_range = attack_range * attack_mask

        # "spread" each starting attack range value across all the following
        # ones.
        attack_range = self._cumprod_resetting_on_zero(attack_range)

        # use our new range axis to compute the attack slope
        attack_slope = attack_range * attack_decay_axis / attack + (
            1 - attack_range
        )

        return attack_slope
