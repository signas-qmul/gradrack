import pytest
from pytest_mock import mocker

from gradrack.oscillators import Oscillator, Sinusoid


def test_can_instantiate_and_is_oscillator_subclass():
    sine_osc = Sinusoid()
    assert isinstance(sine_osc, Oscillator)
