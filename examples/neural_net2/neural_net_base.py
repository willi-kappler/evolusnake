# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override
import random as rnd
import math

# Local imports:
from evolusnake.es_individual import ESIndividual

from neuron import Neuron
from dataprovider import DataProvider

logger = logging.getLogger(__name__)


class NeuralNetBase(ESIndividual):
    def __init__(self, input_size: int, output_size: int, data_provider: DataProvider,
                 network_size: int = 0, use_softmax: bool = False):
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.data_provider: DataProvider = data_provider
        self.hidden_layer_size: int = 0
        self.use_softmax: bool = use_softmax

        self.es_randomize()

        if network_size > 0:
            for _ in range(network_size):
                self.add_neuron()

    def description(self) -> str:
        return "NeuralNetBase"

    def test_network(self) -> float:
        loss: float = 0.0

        rounds: int = 10  # -> Hyperparameter

        for _ in range(rounds):
            for (input_values, expected_output) in self.data_provider.test_batch():
                loss += self.evaluate_with_error(input_values, expected_output)

        return loss / (self.data_provider.batch_size * rounds)

    def evaluate(self, input_values: list):
        # First reset all values to 0.0:
        for neuron in self.hidden_layer:
            neuron.current_value = 0.0

        for _ in range(2):
            for neuron in self.hidden_layer:
                neuron.evaluate(input_values, self.hidden_layer)

    def calc_error(self, expected_output: list) -> float:
        if self.use_softmax:
            return self.calc_error2(expected_output)
        else:
            return self.calc_error1(expected_output)

    def calc_error1(self, expected_output: list) -> float:
        error: float = 0.0

        # The first neurons are output neurons
        for i in range(self.output_size):
            error += abs(expected_output[i] - self.hidden_layer[i].current_value)

        return error

    def calc_error2(self, expected_output: list) -> float:
        error: float = 0.0

        # Use softmax when training for classification.
        # The first neurons are output neurons
        res: list = [math.exp(self.hidden_layer[i].current_value) for i in range(self.output_size)]
        sum_res: float = sum(res)
        for i in range(self.output_size):
            error += abs(expected_output[i] - (res[i] / sum_res))

        #raise ValueError("Does this work (calc_error2) ?")

        return error

    def evaluate_with_error(self, input_values: list, expected_output: list) -> float:
        self.evaluate(input_values)
        return self.calc_error(expected_output)

    def ab_weight_sum(self) -> float:
        ws = 0.0

        for neuron in self.hidden_layer:
            ws += neuron.abs_weight_sum()

        return ws

    def connections_per_neuron(self) -> float:
        result = 0.0

        for neuron in self.hidden_layer:
            result += neuron.num_of_connections()

        return result / self.hidden_layer_size

    def add_neuron(self):
        new_neuron: Neuron = Neuron()

        # Add a random connection to the neuron:
        index: int = rnd.randrange(self.hidden_layer_size)
        new_neuron.add_hidden_connection(index)

        # Add a connection from a random existing neuron to this new neuron:
        index: int = rnd.randrange(self.hidden_layer_size)
        self.hidden_layer[index].add_hidden_connection(self.hidden_layer_size)

        self.hidden_layer.append(new_neuron)
        self.hidden_layer_size += 1

    def add_input_connection(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        neuron = self.hidden_layer[index]

        new_index:int = rnd.randrange(self.input_size)
        neuron.add_input_connection(new_index)

    def add_hidden_connection(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        neuron = self.hidden_layer[index]

        new_index:int = rnd.randrange(self.hidden_layer_size)
        neuron.add_hidden_connection(new_index)

    def randomize_all_neurons(self):
        for neuron in self.hidden_layer:
            neuron.randomize_all_values()

    def clone_base(self, other):
        other.hidden_layer = [n.clone() for n in self.hidden_layer]
        other.hidden_layer_size = self.hidden_layer_size

        return other

    @override
    def es_randomize(self):
        self.hidden_layer: list[Neuron] = []

        for _ in range(self.output_size):
            self.hidden_layer.append(Neuron())

        prev_size: int = self.hidden_layer_size
        self.hidden_layer_size = self.output_size

        for i in range(self.input_size):
            self.add_neuron()
            self.hidden_layer[-1].add_input_connection(i)

        # Did this node already have a big network ?
        # If yes add the same number of neurons.
        diff: int = prev_size - self.hidden_layer_size

        if diff > 0:
            for _ in range(diff):
                self.add_neuron()

    @override
    def es_calculate_fitness(self):
        current_fitness: float = 0.0

        for (input_values, expected_output) in self.data_provider.training_batch():
            current_fitness += self.evaluate_with_error(input_values, expected_output)

        self.fitness = current_fitness / self.data_provider.batch_size

    @override
    def es_calculate_fitness2(self):
        self.fitness2 = self.test_network()

    @override
    def es_from_server(self, other):
        self.hidden_layer = other.hidden_layer
        self.hidden_layer_size = other.hidden_layer_size
        self.fitness = other.fitness
        self.use_softmax = other.use_softmax

    @override
    def es_to_json(self) -> dict:
        data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_layer_size": self.hidden_layer_size,
            "hidden_layer": [n.to_json() for n in self.hidden_layer]
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.fitness = data["fitness"]
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]

        for n in data["hidden_layer"]:
            neuron = Neuron()
            neuron.from_json(n)
            self.hidden_layer.append(neuron)

        self.hidden_layer_size = len(self.hidden_layer)

    @override
    def es_new_best_individual(self):
        logger.info(f"Fitness1: {self.fitness}, fitness2: {self.fitness2}")
        logger.info(f"Size: {self.hidden_layer_size}, absolute weight sum: {self.ab_weight_sum()}, connections per neuron: {self.connections_per_neuron()}")

