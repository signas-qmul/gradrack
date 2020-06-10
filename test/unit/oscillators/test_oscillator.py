import math

import pytest
from pytest_mock import mocker  # noqa
import torch

from gradrack.oscillators import Oscillator, LengthMismatchError


class TestOscillator:

    # FIXTURES
    @pytest.fixture(autouse=True)
    def dummy_osc(self):
        class DummyOsc(Oscillator):
            """ A dummy Oscillator subclass
            """
            def __init__(self):
                super().__init__()

            def _generate(self, phase):
                return phase

        self.dummy_osc = DummyOsc()

    @pytest.fixture
    def mock_dummy_osc(self, mocker):
        """ Creates a mock of the dummy Oscillator subclass' _generate method
        """
        mocker.patch.object(self.dummy_osc, '_generate')
        return self.dummy_osc

    # TESTS
    def test_cant_instantiate_osc_base_class(self):
        with pytest.raises(TypeError):
            test_osc = Oscillator()
            del test_osc

    def test_derived_subclass_is_torch_nn_module(self):
        assert isinstance(self.dummy_osc, torch.nn.Module)

    def test_forward_calls_subclass_generate_method(self, mock_dummy_osc):
        dummy_freq = torch.Tensor([1])
        dummy_phase_mod = torch.Tensor([0])
        dummy_length = 1
        dummy_sample_rate = 1

        mock_dummy_osc(
            dummy_freq,
            dummy_phase_mod,
            dummy_length,
            dummy_sample_rate)

        mock_dummy_osc._generate.assert_called_once_with(torch.Tensor([0]))

    def test_passes_kwargs_to_generate_method(self, mock_dummy_osc):
        dummy_freq = torch.Tensor([0])
        dummy_length = 1
        dummy_kwargs = {
            'foo': 'bar'
        }

        mock_dummy_osc(dummy_freq, length=dummy_length, **dummy_kwargs)

        mock_dummy_osc._generate.assert_called_once_with(
            torch.Tensor([0]),
            foo='bar'
        )

    def check_computes_correct_phase(
            self,
            mock_dummy_osc,
            dummy_freq,
            dummy_phase_mod,
            dummy_length,
            dummy_sample_rate,
            expected_phase):
        mock_dummy_osc(
            dummy_freq,
            dummy_phase_mod,
            dummy_length,
            dummy_sample_rate)

        args = mock_dummy_osc._generate.call_args[0]
        torch.testing.assert_allclose(args[0], expected_phase)

    def test_converts_scalar_frequency_to_phase(self, mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([1]),
            torch.Tensor([0]),
            4,
            4,
            torch.tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2]))

    def test_offsets_computed_phase_by_scalar_phase_mod(self, mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([2]),
            torch.tensor([500]),
            4,
            4,
            torch.Tensor(
                [500, math.pi + 500, 2 * math.pi + 500, 3 * math.pi + 500]))

    def test_computes_phase_from_frequency_with_time_axis(self,
                                                          mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([0, 1, 2, 0]),
            torch.Tensor([20]),
            None,
            4,
            torch.Tensor(
                [20, 20, math.pi / 2 + 20, 3 * math.pi / 2 + 20]))

    def test_computes_phase_from_phasemod_with_time_axis(self, mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([1]),
            torch.Tensor([100, -100, 100, -100]),
            None,
            4,
            torch.Tensor([
                100 + 0,
                -100 + math.pi / 2,
                100 + math.pi,
                -100 + 3 * math.pi / 2]))

    def test_computes_phase_from_scalar_inputs_across_multiple_batches(
            self,
            mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[1], [2]]),
            torch.Tensor([[100], [-50]]),
            4,
            4,
            torch.Tensor([
                [100 + 0, 100 + math.pi / 2, 100 + math.pi,
                 100 + 3 * math.pi / 2],
                [-50 + 0, -50 + math.pi, -50 + 2 * math.pi,
                 -50 + 3 * math.pi]])
        )

    def test_computes_phase_from_time_axis_inputs_across_multiple_batches(
          self,
          mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[0, 1, 2, 0], [2, 1, 0, 0]]),
            torch.Tensor([[10, -10, 5, -5], [1, 2, 3, 4]]),
            None,
            4,
            torch.Tensor([
                [10 + 0, -10 + 0, 5 + math.pi / 2, -5 + 3 * math.pi / 2],
                [1 + 0, 2 + math.pi, 3 + 3 * math.pi / 2,
                 4 + 3 * math.pi / 2]])
        )

    def test_computes_phase_from_scalar_freq_and_time_axis_phasemod_with_batch(
            self,
            mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[0, 1, 2, 0], [2, 1, 0, 0]]),
            torch.Tensor([[10], [10]]),
            None,
            4,
            torch.Tensor([
                [10 + 0, 10 + 0, 10 + math.pi / 2, 10 + 3 * math.pi / 2],
                [10 + 0, 10 + math.pi, 10 + 3 * math.pi / 2,
                 10 + 3 * math.pi / 2]]
            )
        )

    def test_computes_phase_from_time_axis_freq_and_scalar_phasemod_with_batch(
            self,
            mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[1], [2]]),
            torch.Tensor([[10, -10, 5, -5], [1, 2, 3, 4]]),
            None,
            4,
            torch.Tensor([
                [10 + 0, -10 + math.pi / 2, 5 + math.pi, -5 + 3 * math.pi / 2],
                [1 + 0, 2 + math.pi, 3 + 2 * math.pi, 4 + 3 * math.pi]])
        )

    def test_zero_length_output_when_length_scalar_is_zero(
            self,
            mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([0]),
            None,
            0,
            None,
            torch.Tensor([])
        )

    def test_computes_angular_frequency_when_no_sample_rate_passed(
            self,
            mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[math.pi]]),
            torch.Tensor([[0]]),
            4,
            None,
            torch.Tensor([[0, math.pi, 2 * math.pi, 3 * math.pi]]))

    def test_phase_mod_is_optional(self, mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([0]),
            None,
            1,
            4,
            torch.Tensor([0]))

    def test_phase_mod_is_optional_with_batch_dimension(self, mock_dummy_osc):
        self.check_computes_correct_phase(
            mock_dummy_osc,
            torch.Tensor([[0], [0]]),
            None,
            1,
            4,
            torch.Tensor([[0], [0]]))

    def test_returns_generated_values(self):
        dummy_freq = torch.Tensor([0])
        dummy_length = 2
        expected_output = torch.Tensor([0, 0])

        actual_output = self.dummy_osc(dummy_freq, length=dummy_length)

        torch.testing.assert_allclose(actual_output, expected_output)

    def check_for_length_mismatch_error(
            self,
            mock_dummy_osc,
            dummy_freq,
            dummy_phase_mod,
            dummy_length,
            dummy_sample_rate):
        with pytest.raises(LengthMismatchError):
            mock_dummy_osc(
                dummy_freq,
                dummy_phase_mod,
                length=dummy_length,
                sample_rate=dummy_sample_rate)

    def test_throws_if_no_length_given_when_frequency_and_phase_mod_are_scalar(
           self,
           mock_dummy_osc):
        self.check_for_length_mismatch_error(
            mock_dummy_osc,
            torch.Tensor([1]),
            torch.Tensor([0]),
            None,
            4)

    def test_throws_if_length_given_when_frequency_time_axis_used(
            self,
            mock_dummy_osc):
        self.check_for_length_mismatch_error(
            mock_dummy_osc,
            torch.Tensor([1, 2, 3]),
            torch.Tensor([0]),
            4,
            4)

    def test_throws_if_length_given_when_phase_time_axis_used(self,
                                                              mock_dummy_osc):
        self.check_for_length_mismatch_error(
            mock_dummy_osc,
            torch.Tensor([0]),
            torch.Tensor([1, 2, 3]),
            4,
            4)

    def test_throws_if_length_given_when_freq_time_axis_used_and_phase_unspec(
            self,
            mock_dummy_osc):
        self.check_for_length_mismatch_error(
            mock_dummy_osc,
            torch.Tensor([1, 2, 3]),
            None,
            4,
            4)
