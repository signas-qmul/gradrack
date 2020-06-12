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
            torch.Tensor([1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None,
            torch.Tensor([0.25, 0.5, 0.75, 1]))

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
            torch.Tensor([1, 1, 1, 1, 1]),
            torch.Tensor([4]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None,
            torch.Tensor([0.25, 0.5, 0.75, 1, 1 / math.e])
        )

    def test_generates_multi_sample_decay_with_non_zero_sustain(self):
        dummy_decay = 4
        dummy_sustain = 0.5
        dummy_time_constant = 1 / math.e ** (1 / dummy_decay)
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 1, 1, 1, 1, 1]),
            torch.Tensor([2]),
            torch.Tensor([dummy_decay]),
            torch.Tensor([dummy_sustain]),
            torch.Tensor([1]),
            None,
            torch.Tensor([
                0.5,
                1,
                dummy_sustain + (1 - dummy_sustain) * dummy_time_constant ** 1,
                dummy_sustain + (1 - dummy_sustain) * dummy_time_constant ** 2,
                dummy_sustain + (1 - dummy_sustain) * dummy_time_constant ** 3,
                dummy_sustain + (1 - dummy_sustain) * dummy_time_constant ** 4,
                dummy_sustain + (1 - dummy_sustain) * dummy_time_constant ** 5
            ])
        )

    def test_generates_single_sample_release_from_full_sustain(self):
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 0]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            None,
            torch.Tensor([1, 1, 1 / math.e])
        )

    def test_generates_multi_sample_release_from_mid_decay_portion(self):
        dummy_release = 4
        dummy_time_constant = 1 / math.e ** (1 / dummy_release)
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([0]),
            torch.Tensor([dummy_release]),
            None,
            torch.Tensor([
                0.5,
                1,
                1 / math.e ** 0.5,
                1 / math.e ** 1,
                (1 / math.e) * dummy_time_constant ** 1,
                (1 / math.e) * dummy_time_constant ** 2,
                (1 / math.e) * dummy_time_constant ** 3,
                (1 / math.e) * dummy_time_constant ** 4
            ])
        )

    def test_generates_multiple_envelopes_from_retriggering_gate(self):
        self.check_correctly_generates_envelope(
            torch.Tensor([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
            torch.Tensor([2]),
            torch.Tensor([2]),
            torch.Tensor([0]),
            torch.Tensor([2]),
            None,
            torch.Tensor([
                0,
                0.5,
                1.0,
                (1 / math.e ** 0.5) ** 1,
                (1 / math.e ** 0.5) ** 2,
                (1 / math.e) * (1 / math.e ** 0.5),
                (1 / math.e) * (1 / math.e),
                0.1353352832 + (1 - 0.1353352832) / 2,
                1.0,
                (1 / math.e ** 0.5) ** 1,
                (1 / math.e ** 0.5) ** 2,
                (1 / math.e) * (1 / math.e ** 0.5) ** 1,
                (1 / math.e) * (1 / math.e ** 0.5) ** 2
            ])
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

    def test_throws_when_attack_is_zero(self):
        self.check_throws_value_error(
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            None)
