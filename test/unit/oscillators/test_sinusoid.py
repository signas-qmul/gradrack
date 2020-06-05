import math

import pytest
from pytest_mock import mocker
import torch

from gradrack.oscillators import Oscillator, Sinusoid


@pytest.fixture
def sine_osc():
    return Sinusoid()


def test_can_instantiate_and_is_oscillator_subclass(sine_osc):
    assert isinstance(sine_osc, Oscillator)


def check_computes_correct_signal(
        sine_osc,
        dummy_freq,
        dummy_phase_mod,
        dummy_length,
        dummy_sr,
        expected_output):
    actual_output = sine_osc(
        dummy_freq,
        phase_mod=dummy_phase_mod,
        length=dummy_length,
        sample_rate=dummy_sr)
    torch.testing.assert_allclose(actual_output, expected_output)


def test_outputs_zero_at_phase_zero(sine_osc):
    check_computes_correct_signal(
        sine_osc,
        torch.Tensor([0]),
        None,
        1,
        None,
        torch.Tensor([0]))


def test_outputs_sine_wave_at_fixed_frequency(sine_osc):
    check_computes_correct_signal(
        sine_osc,
        torch.Tensor([1]),
        None,
        4,
        8,
        torch.Tensor([0, 1 / math.sqrt(2), 1, 1 / math.sqrt(2)]))


def test_outputs_sine_wave_at_varying_frequency(sine_osc):
    check_computes_correct_signal(
        sine_osc,
        torch.Tensor([0, 1, 2, 1]),
        None,
        None,
        4,
        torch.Tensor([0, 0, 1, -1]))


def test_outputs_sine_wave_with_fixed_phase_mod(sine_osc):
    check_computes_correct_signal(
        sine_osc,
        torch.Tensor([1]),
        torch.Tensor([math.pi / 2]),
        4,
        4,
        torch.Tensor([1, 0, -1, 0]))
