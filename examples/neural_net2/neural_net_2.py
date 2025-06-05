# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override, Self

# Local imports:
import evolusnake.es_utils as utils

from dataprovider import DataProvider
from neural_net_base import NeuralNetBase
from neuron import Neuron

logger = logging.getLogger(__name__)


class NeuralNetIndividual2(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, network_size: int = 0,
                 use_softmax: bool = False, max_size: int = 0):
        super().__init__(input_size, output_size, data_provider, network_size,
                         use_softmax, max_size)

        self.new_fitness_needed: bool = True

    def mutate_bias(self):
        neuron: Neuron = self.get_random_neuron()[0]
        delta: float = utils.es_uniform3()

        self.es_calculate_fitness()
        fitness1: float = self.fitness
        bias1: float = neuron.bias

        neuron.bias = min(1.0, bias1 + delta)
        self.es_calculate_fitness()
        fitness2: float = self.fitness
        bias2: float = neuron.bias

        neuron.bias = max(-1.0, bias1 - delta)
        self.es_calculate_fitness()
        fitness3: float = self.fitness
        bias3: float = neuron.bias

        if fitness2 < fitness1:
            if fitness3 < fitness2:
                neuron.bias = bias3
                self.fitness = fitness3
            else:
                neuron.bias = bias2
                self.fitness = fitness2
        elif fitness3 < fitness1:
            neuron.bias = bias3
            self.fitness = fitness3
        else:
            neuron.bias = bias1
            self.fitness = fitness1

    def mutate_input_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        delta: float = utils.es_uniform3()
        connection: list = neuron.get_random_input_connection()

        if connection:
            self.es_calculate_fitness()
            fitness1: float = self.fitness
            weight1: float = connection[1]

            connection[1] = min(1.0, weight1 + delta)
            self.es_calculate_fitness()
            fitness2: float = self.fitness
            weight2: float = connection[1]

            connection[1] = max(-1.0, weight1 - delta)
            self.es_calculate_fitness()
            fitness3: float = self.fitness
            weight3: float = connection[1]

            if fitness2 < fitness1:
                if fitness3 < fitness2:
                    connection[1] = weight3
                    self.fitness = fitness3
                else:
                    connection[1] = weight2
                    self.fitness = fitness2
            elif fitness3 < fitness1:
                connection[1] = weight3
                self.fitness = fitness3
            else:
                connection[1] = weight1
                self.fitness = fitness1

    def mutate_hidden_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        delta: float = utils.es_uniform3()
        connection: list = neuron.get_random_hidden_connection()

        if connection:
            self.es_calculate_fitness()
            fitness1: float = self.fitness
            weight1: float = connection[1]

            connection[1] = min(1.0, weight1 + delta)
            self.es_calculate_fitness()
            fitness2: float = self.fitness
            weight2: float = connection[1]

            connection[1] = max(-1.0, weight1 - delta)
            self.es_calculate_fitness()
            fitness3: float = self.fitness
            weight3: float = connection[1]

            if fitness2 < fitness1:
                if fitness3 < fitness2:
                    connection[1] = weight3
                    self.fitness = fitness3
                else:
                    connection[1] = weight2
                    self.fitness = fitness2
            elif fitness3 < fitness1:
                connection[1] = weight3
                self.fitness = fitness3
            else:
                connection[1] = weight1
                self.fitness = fitness1

    @override
    def description(self) -> str:
        return ("NeuralNet2: Try two different small mutations and compare them to the current fitness."
            "Use the best one.")

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_bias()
                self.new_fitness_needed = False
            case 1:
                self.mutate_input_connection()
                self.new_fitness_needed = False
            case 2:
                self.mutate_hidden_connection()
                self.new_fitness_needed = False
            case 3:
                if self.common_mutations():
                    self.es_mutate(utils.es_rand_int(3))
                else:
                    self.new_fitness_needed = True
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 2")
                raise ValueError(f"Unknown operation: {mut_op} in net 2")

    @override
    def es_calculate_fitness(self):
        if self.new_fitness_needed:
            super().es_calculate_fitness()

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual2(self.input_size, self.output_size, self.data_provider,
                    self.network_size, self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore
