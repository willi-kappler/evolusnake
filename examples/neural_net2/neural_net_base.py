# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
from typing import override
import math

# Local imports:
from evolusnake.es_individual import ESIndividual
import evolusnake.es_utils as utils

from neuron import Neuron
from dataprovider import DataProvider

logger = logging.getLogger(__name__)


class NeuralNetBase(ESIndividual):
    def __init__(self, input_size: int, output_size: int, data_provider: DataProvider,
                 use_softmax: bool = False, max_size: int = 100):
        super().__init__()

        if output_size > max_size:
            logger.error("output_size > max_size")
            logger.error(f"{output_size=} > {max_size=}")
            raise ValueError(f"output_size ({output_size}) > max_size ({max_size})!")

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.max_size: int = max_size
        self.data_provider: DataProvider = data_provider
        self.hidden_layer_size: int = 0
        self.use_softmax: bool = use_softmax
        self.hidden_layer: list[Neuron] = []
        self.hidden_layer_size: int = output_size

        for _ in range(output_size):
            self.hidden_layer.append(Neuron())

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

    def abs_weight_sum(self) -> float:
        ws = 0.0

        for neuron in self.hidden_layer:
            ws += neuron.abs_weight_sum()

        return ws

    def connections_per_neuron(self) -> float:
        result = 0.0

        for neuron in self.hidden_layer:
            result += neuron.num_of_connections()

        return result / self.hidden_layer_size

    def grow_to_size(self, size: int):
        for _ in range(size):
            self.add_neuron()

    def mutate_bias1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_bias1()

    def mutate_bias2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_bias2()

    def mutate_input_connection1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_input_connection1()

    def mutate_input_connection2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_input_connection2()

    def mutate_hidden_connection1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_hidden_connection1()

    def mutate_hidden_connection2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_hidden_connection2()

    def add_neuron(self):
        if self.hidden_layer_size < self.max_size:
            new_neuron: Neuron = Neuron()

            # Add a random connection to the neuron:
            index: int = utils.es_rand_int(self.hidden_layer_size)
            new_neuron.add_hidden_connection(index)

            # Add a connection from a random existing neuron to this new neuron:
            neuron: Neuron = self.get_random_neuron()[0]
            neuron.add_hidden_connection(self.hidden_layer_size)

            self.hidden_layer.append(new_neuron)
            self.hidden_layer_size += 1
        else:
            self.mutate_hidden_connection2()

    def get_random_neuron(self) -> tuple[Neuron, int]:
        index: int = utils.es_rand_int(self.hidden_layer_size)
        return (self.hidden_layer[index], index)

    def add_input_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]

        new_index: int = utils.es_rand_int(self.input_size)
        neuron.add_input_connection(new_index)

    def add_hidden_connection(self):
        neuron: Neuron = self.get_random_neuron()[0]

        new_index: int = self.get_random_neuron()[1]
        neuron.add_hidden_connection(new_index)

    def randomize_all_neurons(self):
        for neuron in self.hidden_layer:
            neuron.randomize_all_values()

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
        (neuron1, index1) = self.get_random_neuron()
        if neuron1.is_empty():
            self.mutate_hidden_connection2()
            return

        (neuron2, index2) = self.get_random_neuron()
        if neuron2.is_empty():
            self.mutate_hidden_connection2()
            return

        while index1 == index2:
            index2 = self.get_random_neuron()[1]

        (self.hidden_layer[index1], self.hidden_layer[index2]) = (self.hidden_layer[index2], self.hidden_layer[index1])

    def prune_connections(self):
        for neuron in self.hidden_layer:
            neuron.prune_connections()

    def change_activation_function(self):
        neuron = self.get_random_neuron()[0]
        neuron.mutate_activation()

    def split_neuron(self):
        if self.hidden_layer_size < self.max_size:
            neuron: Neuron = self.get_random_neuron()[0]
            new_neuron: Neuron = neuron.split_neuron()

            neuron = self.get_random_neuron()[0]
            neuron.add_hidden_connection(self.hidden_layer_size)

            self.hidden_layer.append(new_neuron)
            self.hidden_layer_size += 1
        else:
            self.mutate_hidden_connection2()

    def shuffle_input_connections(self):
        neuron = self.get_random_neuron()[0]
        neuron.shuffle_input_connections()

    def shuffle_hidden_connections(self):
        neuron = self.get_random_neuron()[0]
        neuron.shuffle_hidden_connections()

    def common_mutations(self) -> bool:
        mut_op: int = utils.es_rand_int(13)

        match mut_op:
            case 0:
                self.add_input_connection()
            case 1:
                self.add_hidden_connection()
            case 2:
                self.add_neuron()
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
            case 10:
                self.split_neuron()
            case 11:
                self.shuffle_input_connections()
            case 12:
                self.shuffle_hidden_connections()

        return False

    def clone_base(self, other):
        other.hidden_layer = [n.clone() for n in self.hidden_layer]
        other.hidden_layer_size = self.hidden_layer_size
        other.max_size = self.max_size

        return other

    @override
    def es_randomize(self):
        self.hidden_layer: list[Neuron] = []

        for _ in range(self.hidden_layer_size):
            self.hidden_layer.append(Neuron())

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
        logger.info(f"Size: {self.hidden_layer_size}, absolute weight sum: {self.abs_weight_sum()}, "
             f"connections per neuron: {self.connections_per_neuron()}")
