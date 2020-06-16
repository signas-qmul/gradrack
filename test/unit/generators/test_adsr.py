import math

import pytest
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
                0.0,
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
                (1 / math.e) * (1 / math.e ** 0.5),
                (1 / math.e) * (1 / math.e)
            ])
        )

    def test_generates_multiple_envelopes_when_release_interrupts_attack(self):
        gate = torch.Tensor([0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])

        attack = 4
        decay = 120
        sustain = 0.2
        release = 4
        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        initial_section = torch.Tensor([0])
        attack_section_1 = self._generate_attack_portion(0, attack, 2)
        release_section_1 = self._generate_exponential_decay(
            attack_section_1[-1],
            alpha_r,
            3)
        attack_section_2 = self._generate_attack_portion(
            release_section_1[-1],
            attack,
            2)
        release_section_2 = self._generate_exponential_decay(
            attack_section_2[-1],
            alpha_r,
            2
        )
        attack_section_3 = self._generate_attack_portion(
            release_section_2[-1],
            attack,
            2)
        expected_output = torch.cat((
            initial_section,
            attack_section_1,
            release_section_1,
            attack_section_2,
            release_section_2,
            attack_section_3))

        self.check_correctly_generates_envelope(
            gate,
            attack,
            decay,
            sustain,
            release,
            None,
            expected_output)

    def test_generates_envelope_when_release_interrupts_longer_attack(self):
        alpha_r = 1 / math.e ** 0.125  # release time constant

        attack_1_final_value = 0.2
        release_1_final_value = attack_1_final_value * alpha_r ** 2
        attack_2_final_value = (release_1_final_value
                                + 8
                                * (1 - release_1_final_value)
                                / 10)
        self.check_correctly_generates_envelope(
            torch.Tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            torch.Tensor([10]),
            torch.Tensor([5125]),
            torch.Tensor([0.3]),
            torch.Tensor([8]),
            None,
            torch.Tensor([
                0.1,  # attack 1 begin
                attack_1_final_value,  # attack_1 end
                attack_1_final_value * alpha_r ** 1,  # release 1 begin
                release_1_final_value,  # release 1 end
                #  attack 2 begin:
                release_1_final_value + 1 * (1 - release_1_final_value) / 10,
                release_1_final_value + 2 * (1 - release_1_final_value) / 10,
                release_1_final_value + 3 * (1 - release_1_final_value) / 10,
                release_1_final_value + 4 * (1 - release_1_final_value) / 10,
                release_1_final_value + 5 * (1 - release_1_final_value) / 10,
                release_1_final_value + 6 * (1 - release_1_final_value) / 10,
                release_1_final_value + 7 * (1 - release_1_final_value) / 10,
                attack_2_final_value,  # attack 2 end
                attack_2_final_value * alpha_r ** 1,  # release 2 begin
                attack_2_final_value * alpha_r ** 2,
                attack_2_final_value * alpha_r ** 3,
                attack_2_final_value * alpha_r ** 4  # release 2 end
            ])
        )

    def test_converts_from_seconds_to_samples_when_sample_rate_given(self):
        alpha_d = 1 / math.e ** 0.5  # expected decay time constant
        alpha_r = 1 / math.e ** 0.25  # expected releae time constant
        decay_final_value = alpha_d ** 2

        self.check_correctly_generates_envelope(
            torch.Tensor([0, 1, 1, 1, 1, 0, 0]),
            torch.Tensor([0.5]),
            torch.Tensor([0.5]),
            torch.Tensor([0.0]),
            torch.Tensor([1]),
            4,
            torch.Tensor([
                0,
                0.5,  # attack begin
                1.0,  # attack end
                alpha_d ** 1,  # decay begin
                alpha_d ** 2,  # decay end
                decay_final_value * alpha_r ** 1,  # release begin
                decay_final_value * alpha_r ** 2   # release end
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

    def _compute_time_constants(self, decay, release):
        return (self._compute_time_constant(decay),
                self._compute_time_constant(release))

    def _compute_time_constant(self, length):
        return 1 / math.e ** (1 / length)

    def _generate_attack_portion(self, start, attack_time, length):
        axis = torch.arange(length) + 1
        ramp = (1 - start) * axis / float(attack_time) + start
        return ramp

    def _generate_exponential_decay(self, start, time_constant, length):
        axis = torch.ones(length) * time_constant
        axis[0] *= start
        slope = axis.cumprod(-1)
        return slope
