import pytest
import torch

from gradrack.oscillators import Oscillator


# FIXTURES
@pytest.fixture
def dummy_osc():
    class DummyOsc(Oscillator):
        def __init__(self):
            super().__init__()

        def other_method(self):
            pass
    return DummyOsc()  # noqa: F841


def test_cant_instantiate_osc_base_class():
    with pytest.raises(TypeError):
        test_osc = Oscillator()
        del test_osc


def test_derived_subclass_is_torch_nn_module(dummy_osc):
    assert isinstance(dummy_osc, torch.nn.Module)


def test_has_forward_with_correct_parameters(dummy_osc):
    dummy_freq = torch.Tensor([5])
    dummy_phase = torch.Tensor([-4])
    dummy_length = 24
    dummy_sample_rate = 12

    test_forward = dummy_osc(
        dummy_freq,
        dummy_phase,
        dummy_length,
        dummy_sample_rate)
    del test_forward
