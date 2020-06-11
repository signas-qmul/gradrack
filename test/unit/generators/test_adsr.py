import math

import pytest
from pytest_mock import mocker
import torch

from gradrack.generators import ADSR


class TestADSREnvelope:
    @pytest.fixture(autouse=True)
    def set_adsr(self):
        self.adsr = ADSR()

    def test_is_torch_nn_module(self):
        assert isinstance(self.adsr, torch.nn.Module)

    def check_correctly_generates_envelope(
            self,
            dummy_gate,
            dummy_attack,
            dummy_decay,
            dummy_sustain,
            dummy_release,
            dummy_sample_rate,
            expected_output):
        actual_output = self.adsr(
            dummy_gate,
            dummy_attack,
            dummy_decay,
            dummy_sustain,
            dummy_release,
            dummy_sample_rate)
        torch.testing.assert_allclose(actual_output, expected_output)

    def test_generates_attack_portion(self):
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None,
            torch.Tensor([0, 0.25, 0.5, 0.75, 1]))

    def test_generates_nothing_with_no_gate(self):
        self.check_correctly_generates_envelope(
            torch.Tensor([0, 0, 0]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None,
            torch.Tensor([0, 0, 0]))

    def test_generates_single_sample_decay(self):
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None,
            torch.Tensor([0, 0.25, 0.5, 0.75, 1, 1 / math.e])
        )

    def check_throws_value_error(
            self,
            dummy_gate,
            dummy_attack,
            dummy_decay,
            dummy_sustain,
            dummy_release,
            dummy_sample_rate):
        with pytest.raises(ValueError):
            self.adsr(
                dummy_gate,
                dummy_attack,
                dummy_decay,
                dummy_sustain,
                dummy_release,
                dummy_sample_rate)

    def test_throws_when_zero_decay_is_given(self):
        self.check_throws_value_error(
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([0]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None)

    def test_throws_when_zero_release_is_given(self):
        self.check_throws_value_error(
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([0]),
            None)

    def test_throws_when_attack_is_negative(self):
        self.check_throws_value_error(
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([-1]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None)
