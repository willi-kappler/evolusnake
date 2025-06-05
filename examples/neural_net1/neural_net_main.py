# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
import pathlib
from typing import override, Self

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer
import evolusnake.es_utils as utils

from neuron import Neuron

logger = logging.getLogger(__name__)


class NeuralNetIndividual(ESIndividual):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.new_node_prob: int = 10000
        self.new_connection_prob: int = 100

        self.es_randomize()

    def test_network(self) -> float:
        loss: float = 0.0

        self.evaluate([0.0, 0.0])
        loss += self.calc_error([0.0])
        self.evaluate([1.0, 0.0])
        loss += self.calc_error([1.0])
        self.evaluate([0.0, 1.0])
        loss += self.calc_error([1.0])
        self.evaluate([1.0, 1.0])
        loss += self.calc_error([0.0])

        return loss / 4.0

    def evaluate(self, input_values: list):
        # First reset all values to 0.0:
        for neuron in self.hidden_layer:
            neuron.current_value = 0.0

        for _ in range(2):
            for neuron in self.hidden_layer:
                neuron.evaluate(input_values, self.hidden_layer)

    def calc_error(self, expected_output: list) -> float:
        error: float = 0.0

        # The first neurons are output neurons
        for i in range(self.output_size):
            error += (expected_output[i] - self.hidden_layer[i].current_value)**2.0

        return error

    def evaluate_with_error(self, values):
        (input_values, expected_output) = values
        self.evaluate(input_values)
        self.fitness += self.calc_error(expected_output)

    def add_neuron(self):
        new_neuron: Neuron = Neuron()

        # Add a random connection to the neuron:
        index: int = utils.es_rand_int(self.hidden_layer_size)
        new_neuron.add_hidden_connection(index)

        # Add a connection from this new neuron to a random existing neuron:
        index: int = utils.es_rand_int(self.hidden_layer_size)
        self.hidden_layer[index].add_hidden_connection(self.hidden_layer_size)

        self.hidden_layer.append(new_neuron)
        self.hidden_layer_size += 1

    def swap_neurons(self):
        i1 = utils.es_rand_int(self.hidden_layer_size)
        i2 = utils.es_rand_int(self.hidden_layer_size)

        if self.hidden_layer[i1].is_empty():
            self.mutate_neuron()
            return

        if self.hidden_layer[i2].is_empty():
            self.mutate_neuron()
            return

        (self.hidden_layer[i1], self.hidden_layer[i2]) = (self.hidden_layer[i2], self.hidden_layer[i1])

    def mutate_neuron(self):
        index1: int = utils.es_rand_int(self.hidden_layer_size)
        mut_op: int = utils.es_rand_int(5)
        neuron: Neuron = self.hidden_layer[index1]

        match mut_op:
            case 0:
                neuron.mutate_bias()
            case 1:
                n = utils.es_rand_int(self.new_connection_prob)

                if n == 0:
                    index2 = utils.es_rand_int(self.input_size)
                    neuron.add_input_connection(index2)
                else:
                    neuron.mutate_input_connection()
            case 2:
                n = utils.es_rand_int(self.new_connection_prob)

                if n == 0:
                    index2: int = utils.es_rand_int(self.hidden_layer_size)
                    neuron.add_hidden_connection(index2)
                else:
                    neuron.mutate_hidden_connection()
            case 3:
                index2 = utils.es_rand_int(self.input_size)
                neuron.replace_input_connection(index2)
            case 4:
                index2: int = utils.es_rand_int(self.hidden_layer_size)
                neuron.replace_hidden_connection(index2)

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                n = utils.es_rand_int(self.new_node_prob)

                if n == 0:
                    self.add_neuron()
                else:
                    self.mutate_neuron()
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

        # Start with two neurons
        self.add_neuron()
        self.add_neuron()
        # With two connection to the first two input layer
        self.hidden_layer[-2].add_input_connection(0)
        self.hidden_layer[-2].add_input_connection(1)
        self.hidden_layer[-1].add_input_connection(0)
        self.hidden_layer[-1].add_input_connection(1)

    @override
    def es_calculate_fitness(self):
        self.fitness: float = 0.0

        self.evaluate_with_error(([0.0, 0.0], [0.0]))
        self.evaluate_with_error(([0.0, 1.0], [1.0]))
        self.evaluate_with_error(([1.0, 0.0], [1.0]))
        self.evaluate_with_error(([1.0, 1.0], [0.0]))

        self.fitness = self.fitness / 4.0

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual(self.input_size, self.output_size)
        clone.hidden_layer = [n.clone() for n in self.hidden_layer]
        clone.hidden_layer_size = self.hidden_layer_size

        return clone  # type: ignore

    @override
    def es_from_server(self, other):
        self.hidden_layer = other.hidden_layer
        self.hidden_layer_size = other.hidden_layer_size

    @override
    def es_to_json(self) -> dict:
        data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_layer": [n.to_json() for n in self.hidden_layer]
        }

        return data

    @override
    def es_from_json(self, data: dict):
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

    data_values = []

    # Just simple XOR
    for _ in range(50):
        data_values.append(([0.0, 0.0], [0.0]))
        data_values.append(([0.0, 1.0], [1.0]))
        data_values.append(([1.0, 0.0], [1.0]))
        data_values.append(([1.0, 1.0], [0.0]))

    ind = NeuralNetIndividual(2, 1)

    config.target_fitness = 0.001

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
        best_ind = server.population[0]
        loss = best_ind.test_network()  # type: ignore
        logger.info(f"Loss: {loss}")
    else:
        print("Create and start node.")
        population = es_select_population(config, ind)
        population.ps_run()


if __name__ == "__main__":
    main()

