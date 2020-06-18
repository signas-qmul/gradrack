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
        expected_output,
    ):
        actual_output = self.adsr(
            dummy_gate,
            dummy_attack,
            dummy_decay,
            dummy_sustain,
            dummy_release,
            dummy_sample_rate,
        )
        torch.testing.assert_allclose(actual_output, expected_output)

    def test_generates_attack_portion(self):
        gate = torch.Tensor([1, 1, 1, 1])
        attack = 4
        decay = 1
        sustain = 0
        release = 1

        expected_output = self._generate_attack_portion(0, attack, 4)

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_nothing_with_no_gate(self):
        gate = torch.Tensor([0, 0, 0])
        attack = 4
        decay = 1
        sustain = 0
        release = 1

        expected_output = torch.Tensor([0, 0, 0])

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_single_sample_decay(self):
        gate = torch.Tensor([1, 1, 1, 1, 1])
        attack = 4
        decay = 1
        sustain = 0
        release = 1

        alpha_d, _ = self._compute_time_constants(decay, release)

        attack_section = self._generate_attack_portion(0, attack, 4)
        decay_section = self._generate_exponential_decay(1, alpha_d, 1)
        expected_output = torch.cat((attack_section, decay_section))

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_multi_sample_decay_with_non_zero_sustain(self):
        gate = torch.Tensor([1, 1, 1, 1, 1, 1, 1])
        attack = 2
        decay = 4
        sustain = 0.5
        release = 1

        alpha_d, _ = self._compute_time_constants(decay, release)

        attack_section = self._generate_attack_portion(0, attack, 2)
        decay_section = self._generate_exponential_decay(
            1, alpha_d, 5, sustain
        )
        expected_output = torch.cat((attack_section, decay_section))

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_single_sample_release_from_full_sustain(self):
        gate = torch.Tensor([1, 1, 0])
        attack = 1
        decay = 1
        sustain = 1
        release = 1

        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        attack_section = self._generate_attack_portion(0, attack, 1)
        decay_section = self._generate_exponential_decay(
            1, alpha_d, 1, sustain
        )
        release_section = self._generate_exponential_decay(
            decay_section[-1], alpha_r, 1, 0
        )
        expected_output = torch.cat(
            (attack_section, decay_section, release_section)
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_multi_sample_release_from_mid_decay_portion(self):
        gate = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0])
        attack = 2
        decay = 2
        sustain = 0
        release = 4

        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        attack_section = self._generate_attack_portion(0, attack, 2)
        decay_section = self._generate_exponential_decay(
            1, alpha_d, 2, sustain
        )
        release_section = self._generate_exponential_decay(
            decay_section[-1], alpha_r, 4
        )
        expected_output = torch.cat(
            (attack_section, decay_section, release_section)
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_simple_envelope_across_batches(self):
        gate = torch.Tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0]]
        )
        attack = torch.Tensor([[2], [3]])
        decay = torch.Tensor([[2], [1]])
        sustain = torch.Tensor([[0.5], [0.0]])
        release = torch.Tensor([[4], [2]])

        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        a_attack_section = self._generate_attack_portion(0, attack[0], 2)
        a_decay_section = self._generate_exponential_decay(
            1, alpha_d[0], 2, sustain[0]
        )
        a_release_section = self._generate_exponential_decay(
            a_decay_section[-1], alpha_r[0], 4
        )
        a_expected_output = torch.cat(
            (a_attack_section, a_decay_section, a_release_section)
        )

        b_initial_section = torch.Tensor([0, 0])
        b_attack_section = self._generate_attack_portion(0, attack[1], 3)
        b_decay_section = self._generate_exponential_decay(
            1, alpha_d[1], 1, sustain[1]
        )
        b_release_section = self._generate_exponential_decay(
            b_decay_section[-1], alpha_r[1], 2
        )
        b_expected_output = torch.cat(
            (
                b_initial_section,
                b_attack_section,
                b_decay_section,
                b_release_section,
            )
        )

        expected_output = torch.stack(
            (a_expected_output, b_expected_output), 0
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_multiple_envelopes_from_retriggering_gate(self):
        gate = torch.Tensor([0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        attack = 2
        decay = 2
        sustain = 0
        release = 2

        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        initial_section = torch.Tensor([0])
        attack_section_1 = self._generate_attack_portion(0, attack, 2)
        decay_section_1 = self._generate_exponential_decay(
            attack_section_1[-1], alpha_d, 2, sustain
        )
        release_section_1 = self._generate_exponential_decay(
            decay_section_1[-1], alpha_r, 2
        )
        attack_section_2 = self._generate_attack_portion(
            release_section_1[-1], attack, 2
        )
        decay_section_2 = self._generate_exponential_decay(
            attack_section_2[-1], alpha_d, 2, sustain
        )
        release_section_2 = self._generate_exponential_decay(
            decay_section_2[-1], alpha_r, 2
        )
        expected_output = torch.cat(
            (
                initial_section,
                attack_section_1,
                decay_section_1,
                release_section_1,
                attack_section_2,
                decay_section_2,
                release_section_2,
            )
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
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
            attack_section_1[-1], alpha_r, 3
        )
        attack_section_2 = self._generate_attack_portion(
            release_section_1[-1], attack, 2
        )
        release_section_2 = self._generate_exponential_decay(
            attack_section_2[-1], alpha_r, 2
        )
        attack_section_3 = self._generate_attack_portion(
            release_section_2[-1], attack, 2
        )
        expected_output = torch.cat(
            (
                initial_section,
                attack_section_1,
                release_section_1,
                attack_section_2,
                release_section_2,
                attack_section_3,
            )
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_generates_envelope_when_release_interrupts_longer_attack(self):
        gate = torch.Tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

        attack = 10
        decay = 5125
        sustain = 0.3
        release = 8
        alpha_d, alpha_r = self._compute_time_constants(decay, release)

        attack_section_1 = self._generate_attack_portion(0, attack, 2)
        release_section_1 = self._generate_exponential_decay(
            attack_section_1[-1], alpha_r, 2
        )
        attack_section_2 = self._generate_attack_portion(
            release_section_1[-1], attack, 8
        )
        release_section_2 = self._generate_exponential_decay(
            attack_section_2[-1], alpha_r, 4
        )
        expected_output = torch.cat(
            (
                attack_section_1,
                release_section_1,
                attack_section_2,
                release_section_2,
            )
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, None, expected_output
        )

    def test_converts_from_seconds_to_samples_when_sample_rate_given(self):
        gate = torch.Tensor([0, 1, 1, 1, 1, 0, 0])
        attack = 0.5
        decay = 0.5
        sustain = 0.0
        release = 1.0
        sample_rate = 4

        alpha_d, alpha_r = self._compute_time_constants(
            decay * sample_rate, release * sample_rate
        )

        initial_section = torch.Tensor([0])
        attack_section = self._generate_attack_portion(
            0, attack * sample_rate, 2
        )
        decay_section = self._generate_exponential_decay(
            attack_section[-1], alpha_d, 2, sustain
        )
        release_section = self._generate_exponential_decay(
            decay_section[-1], alpha_r, 2
        )
        expected_output = torch.cat(
            (initial_section, attack_section, decay_section, release_section)
        )

        self.check_correctly_generates_envelope(
            gate, attack, decay, sustain, release, sample_rate, expected_output
        )

    def check_throws_value_error(
        self,
        dummy_gate,
        dummy_attack,
        dummy_decay,
        dummy_sustain,
        dummy_release,
        dummy_sample_rate,
    ):
        with pytest.raises(ValueError):
            self.adsr(
                dummy_gate,
                dummy_attack,
                dummy_decay,
                dummy_sustain,
                dummy_release,
                dummy_sample_rate,
            )

    def test_throws_when_zero_decay_is_given(self):
        gate = torch.Tensor([1, 1, 1, 1, 1, 1])
        attack = 4
        decay = 0
        sustain = 0
        release = 1

        self.check_throws_value_error(
            gate, attack, decay, sustain, release, None,
        )

    def test_throws_when_zero_release_is_given(self):
        gate = torch.Tensor([1, 1, 1, 1, 1, 1])
        attack = 4
        decay = 1
        sustain = 0
        release = 0

        self.check_throws_value_error(
            gate, attack, decay, sustain, release, None,
        )

    def test_throws_when_attack_is_zero(self):
        gate = torch.Tensor([1, 1, 1, 1, 1, 1])
        attack = 0
        decay = 1
        sustain = 0
        release = 1
        self.check_throws_value_error(
            gate, attack, decay, sustain, release, None,
        )

    def _compute_time_constants(self, decay, release):
        return (
            self._compute_time_constant(decay),
            self._compute_time_constant(release),
        )

    def _compute_time_constant(self, length):
        return 1 / math.e ** (1 / length)

    def _generate_attack_portion(self, start, attack_time, length):
        axis = torch.arange(length) + 1
        ramp = (1 - start) * axis / float(attack_time) + start
        return ramp

    def _generate_exponential_decay(self, start, time_constant, length, end=0):
        axis = torch.ones(length) * time_constant
        axis[0] *= start
        slope = axis.cumprod(-1)
        slope = slope * (1 - end) + end
        return slope
