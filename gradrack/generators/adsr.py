import math

import torch
import torch.nn.functional as F


class ADSR(torch.nn.Module):
    """A PyTorch module that generates differentiable ADSR envelopes.

    Given a set of parameters and a gate signal, generates the output of an
    envelope generator using only vectorised CUDA-friendly operations. Uses
    a linear attack stage with exponential decay and release such that the
    decay/release time specifies the time it takes the segment to fall 1 - 1/e
    of the way to its target value.
    """

    def __init__(self):
        """Constructs an ADSR generator
        """
        super().__init__()

    def forward(self, gate, attack, decay, sustain, release, sample_rate=None):
        """Generate an envelope given ADSR parameters and a gate signal.

        Responds to the gate signal by generating an ADSR envelope with the
        given attack, decay, sustain, release parameters. If sample rate is
        not specified, parameters are interpreted in terms of samples. If it
        is, they are interpreted as times in seconds.

        Args:
            gate (torch.Tensor): The sample-wise gate signal used to trigger
                the envelope. A change from zero to one signifies the start of
                an attack portion, and a change from one to zero signifies the
                start of a release portion. Shape can be [T], or [N, T] where
                T is the time dimension and N is the batch dimension.
            attack (torch.Tensor): The attack time of the envelope in either
                seconds or samples (depending on whether the sample_rate is
                given). Can be of shape [1], or [N, 1], where N is the batch
                dimension.
            decay (torch.Tensor): The decay time of the envelope in either
                seconds or samples (depending on whether the sample_rate is
                given). Can be of shape [1], or [N, 1], where N is the batch
                dimension.
            sustain (torch.Tensor): The sustain value of the envelope given as
                a value from zero to one, signifying the proportion of the
                envelope's maximum value at which to sustain.
            release (torch.Tensor): The release time of the envelope in either
                seconds or samples (depending on whether the sample_rate is
                given). Can be of shape [1], or [N, 1], where N is the batch
                dimension.
            sample_rate (float, optional): The sample rate in Hz. If not
                specified, envelope time parameters are interpreted in terms of
                samples. Otherwise, they are interpreted in seconds. Defaults
                to None.
        """
        if sample_rate is not None:
            attack = attack * sample_rate
            decay = decay * sample_rate
            release = release * sample_rate

        attack, decay, sustain, release = self._ensure_tensors(
            attack, decay, sustain, release
        )
        self._validate_input(attack, decay, sustain, release)

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

    def backward(self):
        print('sup')
        super().backward()

    def _ensure_tensors(self, attack, decay, sustain, release):
        """Ensure that envelope parameters are in torch.Tensor form.
        """
        return (
            self._ensure_tensor(attack),
            self._ensure_tensor(decay),
            self._ensure_tensor(sustain),
            self._ensure_tensor(release),
        )

    def _ensure_tensor(self, input):
        """Turn a number into a torch.Tensor if it isn't already.
        """
        return (
            torch.Tensor([input])
            if not isinstance(input, torch.Tensor)
            else input
        )

    def _validate_input(self, attack, decay, sustain, release):
        """Ensure no erroneous values are input"""
        if (attack <= 0).any():
            raise ValueError("Attack length must be > 0")
        if (decay <= 0).any():
            raise ValueError("Decay length must be > 0")
        if (sustain < 0).any() or (sustain > 1).any():
            raise ValueError("Sustain must be between 0 and 1")
        if (release <= 0).any():
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
        big_negative_number = -1e10

        log_t = t.clone()
        log_t[t != 0] = torch.log(t[t != 0])
        log_t[t == 0] = big_negative_number
        cum_log_t = self._cumsum_resetting_on_value(log_t, big_negative_number)
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
