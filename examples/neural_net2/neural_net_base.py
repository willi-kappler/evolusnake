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
                 network_size: int = 0, use_softmax: bool = False, max_size: int = 0):
        super().__init__()

        if max_size > 0:
            if output_size + network_size > max_size:
                logger.error("output_size + network_size > max_size")
                logger.error(f"{output_size=} + {network_size=} > {max_size=}")
                raise ValueError(f"output_size ({output_size}) + network_size ({network_size}) > max_size ({max_size})!")

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.network_size: int = network_size
        self.max_size: int = max_size
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
        if self.hidden_layer_size < self.max_size:
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

        new_index: int = rnd.randrange(self.input_size)
        neuron.add_input_connection(new_index)

    def add_hidden_connection(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        neuron = self.hidden_layer[index]

        new_index: int = rnd.randrange(self.hidden_layer_size)
        neuron.add_hidden_connection(new_index)

    def randomize_all_neurons(self):
        for neuron in self.hidden_layer:
            neuron.randomize_all_values()

    def get_random_neuron(self) -> tuple[Neuron, int]:
        index = rnd.randrange(self.hidden_layer_size)
        return (self.hidden_layer[index], index)

    def remove_neuron(self):
        (neuron1, index) = self.get_random_neuron()
        neuron1.remove_all_connections()

        for neuron2 in self.hidden_layer:
            neuron2.remove_connection_to(index)

    def remove_input_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.remove_input_connection()

    def remove_hidden_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.remove_hidden_connection()

    def swap_neurons(self):
        index1: int = rnd.randrange(self.hidden_layer_size)
        if self.hidden_layer[index1].is_empty():
            return

        index2: int = rnd.randrange(self.hidden_layer_size)
        if self.hidden_layer[index2].is_empty():
            return

        while index1 == index2:
            index2: int = rnd.randrange(self.hidden_layer_size)

        (self.hidden_layer[index1], self.hidden_layer[index2]) = (self.hidden_layer[index2], self.hidden_layer[index1])

    def prune_connections(self):
        for neuron in self.hidden_layer:
            neuron.prune_connections()

    def change_activation_function(self):
        raise NotImplementedError()

    def common_mutations(self) -> bool:
        mut_op: int = rnd.randrange(9)

        match mut_op:
            case 0:
                self.add_input_connection()
            case 1:
                self.add_hidden_connection()
            case 2:
                prob: int = rnd.randrange(1000)
                if prob == 0:
                    if self.max_size > 0:
                        if self.hidden_layer_size < self.max_size:
                            self.add_neuron()
                        else:
                            return True
                    else:
                        self.add_neuron()
                else:
                    return True
            case 3:
                self.randomize_all_neurons()
            case 4:
                self.remove_neuron()
            case 5:
                self.remove_input_connection()
            case 6:
                self.remove_hidden_connection()
            case 7:
                self.swap_neurons()
            case 8:
                self.prune_connections()
            case 9:
                self.change_activation_function()

        return False

    def clone_base(self, other):
        other.hidden_layer = [n.clone() for n in self.hidden_layer]
        other.hidden_layer_size = self.hidden_layer_size
        other.max_size = self.max_size

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
        if self.fitness < 0.01:
            self.fitness2 = self.test_network()

    @override
    def es_from_server(self, other):
        self.hidden_layer = other.hidden_layer
        self.hidden_layer_size = other.hidden_layer_size
        self.max_size = other.max_size
        self.fitness = other.fitness
        self.use_softmax = other.use_softmax

    @override
    def es_to_json(self) -> dict:
        data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "max_size": self.max_size,
            "hidden_layer_size": self.hidden_layer_size,
            "hidden_layer": [n.to_json() for n in self.hidden_layer]
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.fitness = data["fitness"]
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.max_size = data["max_size"]

        for n in data["hidden_layer"]:
            neuron = Neuron()
            neuron.from_json(n)
            self.hidden_layer.append(neuron)

        self.hidden_layer_size = len(self.hidden_layer)

    @override
    def es_new_best_individual(self):
        logger.info(f"Fitness1: {self.fitness}, fitness2: {self.fitness2}")
        logger.info(f"Size: {self.hidden_layer_size}, absolute weight sum: {self.ab_weight_sum()}, connections per neuron: {self.connections_per_neuron()}")

