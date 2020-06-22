import pytest
from pytest_mock import mocker
import torch

from gradrack.generators import ADSR
from gradrack.oscillators import Sinusoid
from gradrack.synths import FMSynth


class TestFMSynth:
    def test_can_construct_with_correct_interface(self):
        operators = (Sinusoid(), Sinusoid(), Sinusoid(), Sinusoid())
        envelope_generators = (ADSR(), ADSR(), ADSR(), ADSR())
        operator_routing = ((3, 0), (2, 1), (1, 0))
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        del fm_synth

    def test_can_forward_with_correct_interface(self):
        operators = (Sinusoid(), Sinusoid())
        operator_routing = ((1, 0),)
        envelope_generators = (ADSR(), ADSR())
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        frequency = torch.Tensor([1])
        ratios = (1, 2)

        a_0, d_0, s_0, r_0 = 0.3, 0.2, 0.5, 0.1
        a_1, d_1, s_1, r_1 = 0.3, 0.2, 0.5, 0.1
        fm_synth(
            gate,
            frequency,
            ratios,
            ((a_0, d_0, s_0, r_0), (a_1, d_1, s_1, r_1)),
        )

    def test_envelope_generator_called_with_correct_params(self, mocker):
        operators = (Sinusoid(),)

        dummy_adsr = ADSR()
        mocker.patch.object(
            dummy_adsr, "forward", return_value=torch.Tensor([1, 2, 3, 4])
        )
        envelope_generators = (dummy_adsr,)

        operator_routing = ()
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([0, 1, 0, 1])
        frequency = torch.Tensor([1000])
        ratios = (1,)

        a, d, s, r = 0.1, 0.2, 0.3, 0.4
        fm_synth(gate, frequency, ratios, ((a, d, s, r),))

        dummy_adsr.forward.assert_called_once_with(
            gate, a, d, s, r, sample_rate
        )

    def test_oscillator_called_with_correct_params_when_freq_is_scalar(
        self, mocker
    ):
        dummy_operator = Sinusoid()
        mocker.patch.object(
            dummy_operator, "forward", return_value=torch.Tensor([4, 1, 2, 3])
        )
        operators = (dummy_operator,)
        envelope_generators = (ADSR(),)
        operator_routing = ()
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([1, 1, 1, 1])
        frequency = torch.Tensor([1234])
        ratios = (1,)

        a, d, s, r = 0.9, 0.8, 0.7, 0.6
        fm_synth(gate, frequency, ratios, ((a, d, s, r),))

        args = dummy_operator.forward.call_args[0]
        kwargs = dummy_operator.forward.call_args[1]

        torch.testing.assert_allclose(
            args[0], frequency.repeat_interleave(gate.shape[-1], dim=-1)
        )
        assert kwargs["sample_rate"] == sample_rate
        assert kwargs["phase_mod"] is None

    def test_oscillator_called_with_correct_params_when_freq_is_tensor(
        self, mocker
    ):
        dummy_operator = Sinusoid()
        mocker.patch.object(
            dummy_operator,
            "forward",
            return_value=torch.Tensor([1, 2, 3, 4, 5]),
        )
        operators = (dummy_operator,)
        envelope_generators = (ADSR(),)
        operator_routing = ()
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([1, 0, 1, 1, 0])
        frequency = torch.Tensor([1234, 1234, 1234, 4321, 0])
        ratios = (1,)

        a, d, s, r = 0.9, 0.8, 0.7, 0.6
        fm_synth(gate, frequency, ratios, ((a, d, s, r),))

        torch.testing.assert_allclose(
            dummy_operator.forward.call_args[0][0], frequency
        )

    def test_oscillator_frequency_adjusted_by_ratio(self, mocker):
        operators = (Sinusoid(), Sinusoid())
        op0_dummy_return = torch.Tensor([1, 2, 3, 4])
        op1_dummy_return = torch.Tensor([4, 3, 2, 1])

        mocker.patch.object(
            operators[0], "forward", return_value=op0_dummy_return
        )
        mocker.patch.object(
            operators[1], "forward", return_value=op1_dummy_return
        )

        envelope_generators = (ADSR(), ADSR())
        operator_routing = ()
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([1, 1, 1, 1])
        frequency = torch.Tensor([1, 1, 1, 1])
        ratios = (1, 4)

        a, d, s, r = 0.9, 0.8, 0.7, 0.6
        fm_synth(gate, frequency, ratios, ((a, d, s, r), (a, d, s, r)))

        torch.testing.assert_allclose(
            operators[0].forward.call_args[0][0], frequency * ratios[0],
        )
        torch.testing.assert_allclose(
            operators[1].forward.call_args[0][0], frequency * ratios[1],
        )

    def test_operators_correctly_chained_in_simple_config(self, mocker):
        op0_dummy_return = torch.Tensor([1, 2, 3, 4])
        op1_dummy_return = torch.Tensor([4, 3, 2, 1])

        operators = (Sinusoid(), Sinusoid())
        mocker.patch.object(
            operators[0], "forward", return_value=op0_dummy_return
        )
        mocker.patch.object(
            operators[1], "forward", return_value=op1_dummy_return
        )

        eg0_dummy_return = torch.Tensor([10, 10, 10, 10])
        eg1_dummy_return = torch.Tensor([-1, -1, -1, -1])
        envelope_generators = (ADSR(), ADSR())
        mocker.patch.object(
            envelope_generators[0], "forward", return_value=eg0_dummy_return,
        )
        mocker.patch.object(
            envelope_generators[1], "forward", return_value=eg1_dummy_return,
        )

        operator_routing = ((1, 0),)
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([1, 1, 1, 1])
        frequency = torch.Tensor([1, 1, 1, 1])
        ratios = (1, 4)
        op_gains = (torch.Tensor([0.5, 0.5, 0.5, 0.5]), 0.8)

        a, d, s, r = 0.9, 0.8, 0.7, 0.6

        expected_output = eg0_dummy_return * op0_dummy_return * op_gains[0]
        actual_output = fm_synth(
            gate, frequency, ratios, ((a, d, s, r), (a, d, s, r)), op_gains
        )

        op0_kwargs = operators[0].forward.call_args[1]

        torch.testing.assert_allclose(
            op0_kwargs["phase_mod"],
            op1_dummy_return * eg1_dummy_return * op_gains[1],
        )

        torch.testing.assert_allclose(actual_output, expected_output)

    def test_operators_correctly_chained_in_complex_config(self, mocker):
        op_dummy_returns = [torch.Tensor([n, n, n]) for n in range(5)]
        operators = [Sinusoid() for _ in range(5)]
        for dummy_return, operator in zip(op_dummy_returns, operators):
            mocker.patch.object(operator, "forward", return_value=dummy_return)

        eg_dummy_returns = [torch.Tensor([0, 1, 0]) for _ in range(5)]
        envelope_generators = [ADSR() for _ in range(5)]
        for dummy_return, eg in zip(eg_dummy_returns, envelope_generators):
            mocker.patch.object(eg, "forward", return_value=dummy_return)

        operator_routing = ((1, 0), (2, 0), (3, 2), (3, 1), (1, 4))
        sample_rate = 44100

        fm_synth = FMSynth(
            operators, envelope_generators, operator_routing, sample_rate
        )

        gate = torch.Tensor([1, 1, 1])
        frequency = torch.Tensor([1])

        ratios = (1, 2, 3.5, 0.1, 2.2)

        a, d, s, r = 0.9, 0.8, 0.7, 0.6

        expected_output = (
            eg_dummy_returns[0] * op_dummy_returns[0]
            + eg_dummy_returns[4] * op_dummy_returns[4]
        )

        actual_output = fm_synth(gate, frequency, ratios, [(a, d, s, r)] * 5)

        op_kwargs = [operator.forward.call_args[1] for operator in operators]
        torch.testing.assert_allclose(
            op_kwargs[0]["phase_mod"],
            (
                eg_dummy_returns[1] * op_dummy_returns[1]
                + eg_dummy_returns[2] * op_dummy_returns[2]
            ),
        )
        torch.testing.assert_allclose(
            op_kwargs[1]["phase_mod"],
            (eg_dummy_returns[3] * op_dummy_returns[3]),
        )
        torch.testing.assert_allclose(
            op_kwargs[2]["phase_mod"],
            (eg_dummy_returns[3] * op_dummy_returns[3]),
        )
        torch.testing.assert_allclose(
            op_kwargs[4]["phase_mod"],
            (eg_dummy_returns[1] * op_dummy_returns[1]),
        )
        torch.testing.assert_allclose(actual_output, expected_output)
