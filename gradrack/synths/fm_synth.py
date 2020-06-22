import torch


class FMSynth(torch.nn.Module):
    def __init__(
        self,
        operators,
        envelope_generators,
        operator_routing,
        sample_rate=None,
    ):
        super().__init__()
        self.operators = operators
        self.envelope_generators = envelope_generators
        self.sample_rate = sample_rate

        self.operator_modulation_sources = [[] for _ in self.operators]
        for routing in operator_routing:
            self.operator_modulation_sources[routing[1]].append(routing[0])

        self.terminal_operators = self._find_terminal_operators(
            self.operator_modulation_sources
        )

    def forward(self, gate, frequency, ratios, eg_params, operator_gains=None):
        operator_outputs = [None] * len(self.operators)
        envelopes = []

        if frequency.shape[-1] == 1:
            frequency = frequency.repeat_interleave(gate.shape[-1], dim=-1)

        for n, eg in enumerate(self.envelope_generators):
            envelopes.append(eg(gate, *eg_params[n], self.sample_rate))

        def render_operator(n):
            if operator_outputs[n] is None:
                mod_sources = [
                    render_operator(m)
                    for m in self.operator_modulation_sources[n]
                ]

                if len(mod_sources) > 0:
                    phase_mod = torch.stack(mod_sources, dim=0).sum(dim=0)
                else:
                    phase_mod = None

                if operator_gains is not None:
                    gain = operator_gains[n]
                else:
                    gain = 1.0

                operator_outputs[n] = (
                    self.operators[n](
                        frequency * ratios[n],
                        sample_rate=self.sample_rate,
                        phase_mod=phase_mod,
                    )
                    * envelopes[n]
                    * gain
                )

            return operator_outputs[n]

        output_signal = torch.stack(
            [
                render_operator(terminal_operator)
                for terminal_operator in self.terminal_operators
            ],
            dim=0,
        ).sum(dim=0)

        return output_signal

    def _find_terminal_operators(self, operator_modulation_sources):
        terminal_operators = []
        for i, _ in enumerate(operator_modulation_sources):
            is_terminal_operator = True
            for j, _ in enumerate(operator_modulation_sources):
                if i == j:
                    continue

                if i in operator_modulation_sources[j]:
                    is_terminal_operator = False
                    break

            if is_terminal_operator:
                terminal_operators.append(i)

        return terminal_operators
