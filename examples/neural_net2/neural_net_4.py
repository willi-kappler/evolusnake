# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override, Self

# External libraries:
import fastrand

# Local imports:
from dataprovider import DataProvider
from neural_net_base import NeuralNetBase

logger = logging.getLogger(__name__)


class NeuralNetIndividual4(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, network_size: int = 0,
                 use_softmax: bool = False, max_size: int = 0):
        super().__init__(input_size, output_size, data_provider, network_size,
                         use_softmax, max_size)

    def mutate_all_neurons(self):
        for neuron in self.hidden_layer:
            neuron.mutate_all_values()

    def change_all_deltas(self):
        for neuron in self.hidden_layer:
            neuron.change_all_deltas()

    def change_some_deltas(self):
        index: int = fastrand.pcg32bounded(self.hidden_layer_size)
        neuron = self.hidden_layer[index]

        neuron.mutate_bias3()
        neuron.mutate_input_connection3()
        neuron.mutate_hidden_connection3()

    @override
    def description(self) -> str:
        return "NeuralNet4: Use swarm."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_all_neurons()
            case 1:
                if self.common_mutations():
                    self.es_mutate(0)
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 4")
                raise ValueError(f"Unknown operation: {mut_op} in net 4")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual4(self.input_size, self.output_size, self.data_provider,
                    self.network_size, self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore

    @override
    def es_calculate_fitness(self):
        prev_fitness: float = self.fitness
        super().es_calculate_fitness()

        if prev_fitness < self.fitness:
            # Fitness has gotten worse, change direction:
            self.change_some_deltas()
