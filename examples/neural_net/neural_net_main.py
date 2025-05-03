# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
import pathlib
from typing import override, Self
import random as rnd
import math

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer


logger = logging.getLogger(__name__)

class Neuron:
    def __init__(self):
        self.input_connections: list = []
        self.hidden_connections: list = []
        self.bias: float = rnd.uniform(-1.0, 1.0)
        self.current_value: float = 0.0

    def change_bias(self):
        self.bias = rnd.uniform(-1.0, 1.0)

    def add_input_connection(self, index: int):
        if index not in self.input_connections:
            weight: float = rnd.random()
            self.input_connections.append([index, weight])
        else:
            self.input_connections[index][1] = rnd.uniform(-1.0, 1.0)

    def remove_input_connection(self):
        l: int = len(self.input_connections)

        if l == 0:
            return
        else:
            index: int = rnd.randrange(l)
            self.input_connections.pop(index)

    def change_input_connection(self):
        l: int = len(self.input_connections)

        if l == 0:
            return
        else:
            index: int = rnd.randrange(l)
            self.input_connections[index][1] = rnd.uniform(-1.0, 1.0)

    def add_hidden_connection(self, index: int):
        if index not in self.hidden_connections:
            weight: float = rnd.random()
            self.hidden_connections.append([index, weight])
        else:
            self.hidden_connections[index][1] = rnd.uniform(-1.0, 1.0)

    def remove_hidden_connection(self):
        l: int = len(self.hidden_connections)

        if l == 0:
            return
        else:
            index: int = rnd.randrange(l)
            self.hidden_connections.pop(index)

    def change_hidden_connection(self):
        l: int = len(self.hidden_connections)

        if l == 0:
            return
        else:
            index: int = rnd.randrange(l)
            self.hidden_connections[index][1] = rnd.uniform(-1.0, 1.0)

    def evaluate(self, input_values: list, hidden_layer: list):
        new_value: float = self.bias

        for (index, weight) in self.input_connections:
            new_value += weight * input_values[index]

        for (index, weight) in self.hidden_connections:
            l: int = len(hidden_layer)
            if index >= l:
                logger.debug(f"index out of range: {index}, length: {l}")
            else:
                new_value += weight * hidden_layer[index].current_value

        # ReLU
        # self.current_value = max(1.0, max(0.0, new_value))
        self.current_value = max(0.0, new_value)

    def to_json(self) -> dict:
        data = {
            "input_connections": self.input_connections,
            "hidden_connections": self.hidden_connections,
            "bias": self.bias,
        }

        return data

    def from_json(self, data: dict):
        self.input_connections = data["input_connections"]
        self.hidden_connections = data["hidden_connections"]
        self.bias = data["bias"]

class NeuralNetIndividual(ESIndividual):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.es_randomize()

    def evaluate(self, input_values: list):
        for _ in range(2):
            for neuron in self.hidden_layer:
                neuron.evaluate(input_values, self.hidden_layer)

    def calc_error(self, expected_output: list) -> float:
        error: float = 0.0

        # The first neurons are output neurons
        for i in range(self.output_size):
            error += abs(expected_output[i] - self.hidden_layer[i].current_value)

        return error

    def add_neuron(self):
        n = rnd.randrange(1000)

        if n == 0:
            self.hidden_layer.append(Neuron())
            self.hidden_layer_size += 1
            logger.debug(f"{self.hidden_layer_size=}")
        else:
            self.mutate_neuron()

    def swap_neurons(self):
        i1 = rnd.randrange(self.hidden_layer_size)
        i2 = rnd.randrange(self.hidden_layer_size)

        (self.hidden_layer[i1], self.hidden_layer[i2]) = (self.hidden_layer[i2], self.hidden_layer[i1])

    def mutate_neuron(self):
        index1: int = rnd.randrange(self.hidden_layer_size)
        mut_op: int = rnd.randrange(7)
        neuron = self.hidden_layer[index1]

        match mut_op:
            case 0:
                neuron.change_bias()
            case 1:
                index2 = rnd.randrange(self.input_size)
                neuron.add_input_connection(index2)
            case 2:
                n = rnd.randrange(100)

                if n == 0:
                    neuron.remove_input_connection()
                else:
                    self.mutate_neuron()
            case 3:
                neuron.change_input_connection()
            case 4:
                index2: int = rnd.randrange(self.hidden_layer_size)
                neuron.add_hidden_connection(index2)
            case 5:
                n = rnd.randrange(100)

                if n == 0:
                    neuron.remove_hidden_connection()
                else:
                    self.mutate_neuron()
            case 6:
                neuron.change_hidden_connection()

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.add_neuron()
            case 1:
                self.swap_neurons()
            case 2:
                self.mutate_neuron()

    @override
    def es_randomize(self):
        self.hidden_layer: list = []

        for _ in range(self.output_size):
            self.hidden_layer.append(Neuron())

        self.hidden_layer_size = self.output_size

    @override
    def es_calculate_fitness(self):
        self.fitness: float = 0.0

        # Simple XOR function
        self.evaluate([0.0, 0.0])
        self.fitness += self.calc_error([0.0])

        self.evaluate([1.0, 0.0])
        self.fitness += self.calc_error([1.0])

        self.evaluate([0.0, 1.0])
        self.fitness += self.calc_error([1.0])

        self.evaluate([1.0, 1.0])
        self.fitness += self.calc_error([0.0])

        self.fitness

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual(self.input_size, self.output_size)
        clone.hidden_layer = self.hidden_layer[:]
        clone.hidden_layer_size = self.hidden_layer_size
        return clone  # type: ignore

    @override
    def es_to_json(self) -> dict:
        data = {
            "fitness": self.fitness,
            "input_size": self.input_size,
            "output_size": self.output_size,
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

    ind = NeuralNetIndividual(2, 1)

    config.target_fitness = 0.0

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
    else:
        print("Create and start node.")
        population = es_select_population(config, ind)
        population.ps_run()


if __name__ == "__main__":
    main()

