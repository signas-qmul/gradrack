import math

import pytest
from pytest_mock import mocker
import torch

from gradrack.oscillators import Oscillator, LengthMismatchError


# FIXTURES
@pytest.fixture
def dummy_osc():
    class DummyOsc(Oscillator):
        def __init__(self):
            super().__init__()

        def generate(self, phase):
            pass
    return DummyOsc()  # noqa: F841

@pytest.fixture
def mock_dummy_osc(dummy_osc, mocker):
    mocker.patch.object(dummy_osc, 'generate')
    return dummy_osc


def test_cant_instantiate_osc_base_class():
    with pytest.raises(TypeError):
        test_osc = Oscillator()
        del test_osc


def test_derived_subclass_is_torch_nn_module(dummy_osc):
    assert isinstance(dummy_osc, torch.nn.Module)


def test_forward_calls_subclass_generate_method(mock_dummy_osc):
    dummy_freq = torch.Tensor([1])
    dummy_phase_mod = torch.Tensor([0])
    dummy_length = 1
    dummy_sample_rate = 1

    mock_dummy_osc(
        dummy_freq,
        dummy_phase_mod,
        dummy_length,
        dummy_sample_rate)

    mock_dummy_osc.generate.assert_called_once_with(torch.Tensor([0]))


def test_converts_scalar_frequency_to_phase(mock_dummy_osc):
    dummy_freq = torch.Tensor([1])
    dummy_phase_mod = torch.Tensor([0])
    dummy_length = 4
    dummy_sample_rate = 4

    expected_phase = torch.tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2])

    mock_dummy_osc(
        dummy_freq,
        dummy_phase_mod,
        dummy_length,
        dummy_sample_rate)

    args = mock_dummy_osc.generate.call_args[0]
    torch.testing.assert_allclose(args[0], expected_phase)


def test_offsets_computed_phase_by_scalar_phase_mod(mock_dummy_osc):
    dummy_freq = torch.Tensor([2])
    dummy_phase_mod = torch.tensor([500])
    dummy_length = 4
    dummy_sample_rate = 4

    expected_phase = torch.Tensor(
        [500, math.pi + 500, 2 * math.pi + 500, 3 * math.pi + 500])
    
    mock_dummy_osc(
        dummy_freq,
        dummy_phase_mod,
        dummy_length,
        dummy_sample_rate)
    
    args = mock_dummy_osc.generate.call_args[0]
    torch.testing.assert_allclose(args[0], expected_phase)


def test_computes_phase_from_frequency_with_time_axis(mock_dummy_osc):
    dummy_freq = torch.Tensor([0, 1, 2, 0])
    dummy_phase_mod = torch.Tensor([20])
    dummy_sample_rate = 4

    expected_phase = torch.Tensor(
        [20, 20, math.pi / 2 + 20, 3 * math.pi / 2 + 20])

    mock_dummy_osc(
        dummy_freq,
        dummy_phase_mod,
        sample_rate=dummy_sample_rate)
    
    args = mock_dummy_osc.generate.call_args[0]
    torch.testing.assert_allclose(args[0], expected_phase)


def test_computes_phase_from_phase_mod_with_time_axis(mock_dummy_osc):
    dummy_freq = torch.Tensor([1])
    dummy_phase_mod = torch.Tensor([100, -100, 100, -100])
    dummy_sample_rate = 4

    expected_phase = torch.Tensor(
        [100 + 0, -100 + math.pi / 2, 100 + math.pi, -100 + 3 * math.pi / 2])
    
    mock_dummy_osc(
        dummy_freq,
        dummy_phase_mod,
        sample_rate=dummy_sample_rate)
    
    args = mock_dummy_osc.generate.call_args[0]
    torch.testing.assert_allclose(args[0], expected_phase)


def test_throws_if_no_length_given_when_frequency_and_phase_mod_are_scalar(
        mock_dummy_osc):
    dummy_freq = torch.Tensor([1])
    dummy_phase_mod = torch.Tensor([0])
    dummy_sample_rate = 4

    with pytest.raises(LengthMismatchError):
        mock_dummy_osc(
            dummy_freq,
            dummy_phase_mod,
            sample_rate=dummy_sample_rate)
