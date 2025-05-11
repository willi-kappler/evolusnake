# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
import pathlib
from typing import override, Self
import random as rnd
from sys import float_info

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer

from neuron import Neuron
from dataprovider import DataProvider, IterationNeural

logger = logging.getLogger(__name__)


def load_data(filename: str) -> list:
    result = []

    min_val: float = float_info.max
    max_val: float = -float_info.max

    with open(filename, "r") as f:
        next(f)  # ignore column names
        for line in f:
            items: list = line.split(",")
            in1: float = float(items[1])
            in2: float = float(items[2])
            in3: float = float(items[3])
            in4: float = float(items[4])
            name: str = items[5].strip()
            kind = None

            match name:
                case "Iris-setosa":
                    kind = [1.0, 0.0, 0.0]
                case "Iris-versicolor":
                    kind = [0.0, 1.0, 0.0]
                case "Iris-virginica":
                    kind = [0.0, 0.0, 1.0]
                case _:
                    raise ValueError(f"Unknown name: '{name}'")

            result.append(([in1, in2, in3, in4], kind))

            min_val = min(min_val, in1, in2, in3, in4)
            max_val = max(max_val, in1, in2, in3, in4)

    max_val -= min_val

    # Normalize input values:
    for (data, _) in result:
        data[0] = (data[0] - min_val) / max_val
        data[1] = (data[1] - min_val) / max_val
        data[2] = (data[2] - min_val) / max_val
        data[3] = (data[3] - min_val) / max_val

    return result


class NeuralNetIndividual(ESIndividual):
    def __init__(self, input_size: int, output_size: int,
            data_provider: DataProvider, network_size: int = 0):
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.data_provider: DataProvider = data_provider
        self.hidden_layer_size: int = 0

        self.es_randomize()

        if network_size > 0:
            for _ in range(network_size):
                self.add_neuron()

    def test_network(self) -> float:
        loss: float = 0.0

        rounds: int = 100  # -> Hyperparameter

        for _ in range(rounds):
            for (input_values, expected_output) in self.data_provider.test_batch():
                loss += self.evaluate_with_error(input_values, expected_output)

        return loss / (self.data_provider.batch_size * rounds)

    def evaluate(self, input_values: list):
        # First reset all values to 0.0:
        for neuron in self.hidden_layer:
            neuron.current_value = 0.0

        for _ in range(3):
            for neuron in self.hidden_layer:
                neuron.evaluate(input_values, self.hidden_layer)

    def calc_error(self, expected_output: list) -> float:
        error: float = 0.0

        # The first neurons are output neurons
        for i in range(self.output_size):
            error += abs(expected_output[i] - self.hidden_layer[i].current_value)

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

        self.hidden_layer.append(new_neuron)
        self.hidden_layer_size += 1

    def remove_neuron(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        self.hidden_layer[index].clear()

        for neuron in self.hidden_layer:
            neuron.remove_neuron_connection(index)

    def swap_neurons(self):
        i1 = rnd.randrange(self.hidden_layer_size)

        if self.hidden_layer[i1].is_empty():
            return

        i2 = rnd.randrange(self.hidden_layer_size)

        while i1 == i2:
            i2 = rnd.randrange(self.hidden_layer_size)

        if self.hidden_layer[i2].is_empty():
            return

        (self.hidden_layer[i1], self.hidden_layer[i2]) = (self.hidden_layer[i2], self.hidden_layer[i1])

    def mutate_neuron(self, neuron: Neuron):
        mut_op: int = rnd.randrange(3)

        match mut_op:
            case 0:
                neuron.mutate_bias()
            case 1:
                neuron.mutate_input_connection()
            case 2:
                neuron.mutate_hidden_connection()

    @override
    def es_mutate(self, mut_op: int):
        index1: int = rnd.randrange(self.hidden_layer_size)
        neuron: Neuron = self.hidden_layer[index1]

        match mut_op:
            case 0:
                prob1: int = rnd.randrange(1000)  # -> Hyperparameter
                if prob1 == 0:
                    self.add_neuron()
                else:
                    self.mutate_neuron(neuron)
            case 1:
                prob2: int = rnd.randrange(10000)  # -> Hyperparameter
                if prob2 == 0:
                    self.remove_neuron()
                else:
                    self.mutate_neuron(neuron)
            case 2:
                self.swap_neurons()
            case 3:
                index3: int = rnd.randrange(self.input_size)
                neuron.add_input_connection(index3)
            case 4:
                index2: int = rnd.randrange(self.hidden_layer_size)
                neuron.add_hidden_connection(index2)
            case 5:
                self.mutate_neuron(neuron)
            case 6:
                index3: int = rnd.randrange(self.input_size)
                neuron.replace_input_connection(index3)
            case 7:
                index2: int = rnd.randrange(self.hidden_layer_size)
                neuron.replace_hidden_connection(index2)
            case 8:
                prob3: int = rnd.randrange(10000)  # -> Hyperparameter
                if prob3 == 0:
                    neuron.remove_input_connection()
                else:
                    self.mutate_neuron(neuron)
            case 9:
                prob3: int = rnd.randrange(10000)  # -> Hyperparameter
                if prob3 == 0:
                    neuron.remove_hidden_connection()
                else:
                    self.mutate_neuron(neuron)

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
        # If yes at least add half the number of neurons.
        diff: int = int((prev_size - self.hidden_layer_size) / 2)

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
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual(self.input_size, self.output_size, self.data_provider)
        clone.hidden_layer = [n.clone() for n in self.hidden_layer]
        clone.hidden_layer_size = self.hidden_layer_size

        return clone  # type: ignore

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
        logger.info(f"Fitness: {self.fitness}")
        logger.info(f"Fitness2: {self.fitness2}")
        logger.info(f"Size: {self.hidden_layer_size}")
        logger.info(f"Absolute weight sum: {self.ab_weight_sum()}")
        logger.info(f"Connections per neuron: {self.connections_per_neuron()}")


def main():
    config = ESConfiguration.from_json("neural_net_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "neural_net_server.log"

    if not server_mode:
        neural_num: int = 1

        while True:
            log_file_name = f"neural_net_node_{neural_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                neural_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    data_values = load_data("Iris.csv")

    dp = DataProvider(data_values, 20)

    ind = NeuralNetIndividual(4, 3, dp, 10)  # -> Hyperparameter

    config.target_fitness = 0.00001
    config.target_fitness2 = 0.05

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
    else:
        print("Create and start node.")
        population = es_select_population(config, ind, IterationNeural())
        population.ps_run()


if __name__ == "__main__":
    main()

