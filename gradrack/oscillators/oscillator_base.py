from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F


class Oscillator(torch.nn.Module, ABC):
    """Abstract base class for creating differentiable oscillators.

    Provides a structure for defining oscillators which will be tracked by
    PyTorch's AutoGrad. Oscillators are created by inheriting from this class
    and implementing the _generate(phase) method. The phase parameter will be
    automatically calculated from the inputs to forward(). It is also important
    to call the superclass constructor with super().__init__() -- this enables
    the oscillator to function as a torch.nn.Module.
    """
    def __init__(self):
        """Constructs an Oscillator.
        """
        super().__init__()

    def forward(
            self,
            frequency,
            phase_mod=None,
            length=None,
            sample_rate=None,
            **_generate_kwargs):
        """Generate a signal given a set of parameters.

        Uses the frequency, phase_mod, length, and sample_rate parameters to
        determine the best output shape, and then generate a signal by calling
        the subclass generate() method. Parameters frequency and phase_mod can
        either be scalar values, or can have time axes and batch dimensions.
        Any extra keyword arguments are passed directly to _generate.

        Args:
            frequency (torch.Tensor): The sample-wise instantaneous frequency
                of the oscillator. Can be of shapes ([S], [N, S], [T], [N, T],
                [..., N, S], [..., N, T]) where S is a scalar value, T is a
                time dimension, N is a batch dimension, and ... signifies any
                number of extra dimensions.
            phase_mod (torch.Tensor, optional): The sample-wise phase
                modulation used in oscillation (useful for pitch modulation
                and FM synthesis). Can be of same shapes as frequency. Shape
                must be broadcastable with frequency. Defaults to None.
            length (int, optional): Length of output's time dimension in
                samples. This is required if a scalar value is passed in for
                both frequency and phase_mod. Otherwise it is inferred from the
                length of the last axis. Defaults to None.
            sample_rate (float, optional): The sample rate in Hz. If not
                specified, instantaneous frequency is interpreted as angular
                frequency in radians per sample. Defaults to None.

        Returns:
            torch.Tensor: The oscillator's output signal.
        """
        phase_mod = self._replace_empty_phase_mod(phase_mod, frequency)
        sample_rate = self._replace_empty_sample_rate(sample_rate)

        self._check_input_shape(frequency, phase_mod, length)
        frequency = self._broadcast_dimensions(frequency, phase_mod, length)

        phase = self._compute_phase(frequency, phase_mod, sample_rate)

        return self._generate(phase, **_generate_kwargs)

    def _replace_empty_phase_mod(self, phase_mod, frequency):
        # If no phase mod is provided, create a zero phase tensor.
        if phase_mod is None:
            phase_mod = torch.zeros_like(frequency)
        return phase_mod

    def _check_input_shape(self, frequency, phase_mod, length):
        # Make sure our input shapes are compatible
        if length is None:
            if frequency.shape[-1] == 1 and phase_mod.shape[-1] == 1:
                raise LengthMismatchError("Sample length must be provided " +
                                          "for scalar frequency and " +
                                          "phase_mod parameters")
        elif frequency.shape[-1] > 1 or phase_mod.shape[-1] > 1:
            raise LengthMismatchError("Can't use length parameter when " +
                                      "a time dimension is provided for " +
                                      "frequency or phase.")

    def _replace_empty_sample_rate(self, sample_rate):
        # If sample rate is empty, set it to 2π
        return sample_rate or math.tau

    def _broadcast_dimensions(self, frequency, phase_mod, length):
        # Expand out our time dimension if necessary, and match everything up
        # before computing phase
        if length is None and phase_mod.shape[-1] > 1:
            length = phase_mod.shape[-1]

        if frequency.shape[-1] == 1:
            frequency =\
                frequency.repeat_interleave(length, dim=-1)

        original_frequency_length = frequency.shape[-1]
        frequency = F.pad(frequency, (1, 0), 'constant', 0)
        frequency = frequency.narrow(-1, 0, original_frequency_length)

        return frequency

    def _compute_phase(self, frequency, phase_mod, sample_rate):
        # Integrate instantaneous frequency along the time axis to get phase
        phase = frequency.cumsum(-1)
        phase = math.tau * phase / sample_rate
        phase = phase + phase_mod
        return phase

    @abstractmethod
    def _generate(self, phase, **kwargs):
        # This should be overridden to create an oscillator
        pass


class LengthMismatchError(Exception):
    pass
