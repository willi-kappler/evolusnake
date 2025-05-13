# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override, Self
import random as rnd

# Local imports:
from dataprovider import DataProvider
from neural_net_base import NeuralNetBase

logger = logging.getLogger(__name__)


class NeuralNetIndividual4(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
            data_provider: DataProvider, network_size: int = 1):
        super().__init__(input_size, output_size, data_provider, network_size)

    def mutate_all_neurons(self):
        for neuron in self.hidden_layer:
            neuron.mutate_all_values()

    def change_all_deltas(self):
        for neuron in self.hidden_layer:
            neuron.change_all_deltas()

    @override
    def description(self) -> str:
        return "NeuralNet4: Use swarm."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_all_neurons()
            case 1:
                prob: int = rnd.randrange(1000)
                if prob == 0:
                    self.add_neuron()
                else:
                    self.es_mutate(0)
            case 2:
                self.add_input_connection()
            case 3:
                self.add_hidden_connection()
            case 4:
                prob: int = rnd.randrange(1000)
                if prob == 0:
                    self.randomize_all_neurons()
                else:
                    self.es_mutate(0)
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 4")
                raise ValueError(f"Unknown operation: {mut_op} in net 4")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual4(self.input_size, self.output_size, self.data_provider)
        clone.hidden_layer = [n.clone() for n in self.hidden_layer]
        clone.hidden_layer_size = self.hidden_layer_size

        return clone  # type: ignore

    @override
    def es_calculate_fitness(self):
        prev_fitness: float = self.fitness
        super().es_calculate_fitness()

        if prev_fitness < self.fitness:
            # Fitness has gotten worse, change direction:
            self.change_all_deltas()

