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


class NeuralNetIndividual5(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, use_softmax: bool = False, max_size: int = 0):
        super().__init__(input_size, output_size, data_provider, use_softmax, max_size)

    def search_bias(self):
        neuron: Neuron = self.get_random_neuron()[0]

        best_bias: float = neuron.bias
        best_fitness: float = self.fitness

        for i in range(11):
            neuron.bias = (i - 5.0) / 5.0
            self.es_calculate_fitness()

            if self.fitness < best_fitness:
                best_fitness = self.fitness
                best_bias = neuron.bias

        neuron.bias = best_bias

    def search_connection(self, connection: list):
        if connection:
            best_weight: float = connection[1]
            best_fitness: float = self.fitness

            for i in range(11):
                connection[1] = (i - 5.0) / 5.0
                self.es_calculate_fitness()

                if self.fitness < best_fitness:
                    best_fitness = self.fitness
                    best_weight = connection[1]

            connection[1] = best_weight

    def search_input_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        connection = neuron.get_random_input_connection()

        self.search_connection(connection)

    def search_hidden_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        connection = neuron.get_random_hidden_connection()

        self.search_connection(connection)

    @override
    def description(self) -> str:
        return "NeuralNet5: Search through parameter space for best value."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_bias2()
            case 1:
                self.mutate_input_connection2()
            case 2:
                self.mutate_hidden_connection2()
            case 3:
                # Hyperparameter: 100
                prob: int = utils.es_rand_int(100)
                if prob == 0:
                    self.search_bias()
                else:
                    self.es_mutate(utils.es_rand_int(3))
            case 4:
                # Hyperparameter: 100
                prob: int = utils.es_rand_int(100)
                if prob == 0:
                    self.search_input_connection()
                else:
                    self.es_mutate(utils.es_rand_int(3))
            case 5:
                # Hyperparameter: 100
                prob: int = utils.es_rand_int(100)
                if prob == 0:
                    self.search_hidden_connection()
                else:
                    self.es_mutate(utils.es_rand_int(3))
            case 6:
                if self.common_mutations():
                    self.es_mutate(utils.es_rand_int(3))

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual5(self.input_size, self.output_size, self.data_provider,
                    self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore
