# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


# Python std lib:
from typing import Self

# Local imports:
import evolusnake.es_utils as utils


def change_delta(value: float) -> float:
    n = utils.es_rand_int(100)
    if n == 0:
        return utils.es_uniform1()
    else:
        delta: float = utils.es_uniform2()
        value += delta
        if value < -1.0:
            return -1.0
        elif value > 1.0:
            return 1.0
        else:
            return value


class Neuron:
    def __init__(self):
        self.input_connections: list = []
        self.hidden_connections: list = []
        self.bias: float = utils.es_uniform1()
        self.current_value: float = 0.0

    def is_empty(self) -> bool:
        return (self.input_connections == []) and (self.hidden_connections == [])

    def mutate_bias(self):
        self.bias = change_delta(self.bias)

    def add_input_connection(self, index: int):
        for (index2, _) in self.input_connections:
            if index == index2:
                self.mutate_input_connection()
                return

        weight: float = utils.es_uniform1()
        self.input_connections.append([index, weight])

    def mutate_input_connection(self):
        n: int = len(self.input_connections)

        if n == 0:
            self.mutate_bias()
            return
        else:
            index: int = utils.es_rand_int(n)
            connection: list = self.input_connections[index]
            connection[1] = change_delta(connection[1])

    def replace_input_connection(self, new_index: int):
        n: int = len(self.input_connections)

        if n == 0:
            self.mutate_bias()
            return
        else:
            index: int = utils.es_rand_int(n)
            connection: list = self.input_connections[index]
            connection[0] = new_index

    def add_hidden_connection(self, index: int):
        for (index2, _) in self.hidden_connections:
            if index == index2:
                self.mutate_hidden_connection()
                return

        weight: float = utils.es_uniform1()
        self.hidden_connections.append([index, weight])

        # logger.debug(f"add_hidden_connection, {index=}")

    def mutate_hidden_connection(self):
        n: int = len(self.hidden_connections)

        if n == 0:
            self.mutate_bias()
            return
        else:
            index: int = utils.es_rand_int(n)
            connection: list = self.hidden_connections[index]
            connection[1] = change_delta(connection[1])

    def replace_hidden_connection(self, new_index: int):
        n: int = len(self.hidden_connections)

        if n == 0:
            self.mutate_bias()
            return
        else:
            index: int = utils.es_rand_int(n)
            connection: list = self.hidden_connections[index]
            connection[0] = new_index
            # logger.debug(f"replace_hidden_connection, {new_index=}")

    def evaluate(self, input_values: list, hidden_layer: list):
        new_value: float = self.bias

        for (index, weight) in self.input_connections:
            new_value += weight * input_values[index]

        for (index, weight) in self.hidden_connections:
            new_value += weight * hidden_layer[index].current_value

        # ReLU
        self.current_value = max(0.0, new_value)

    def clone(self) -> Self:
        n = Neuron()
        # Do proper cloning!
        n.input_connections = [[i, w] for (i, w) in self.input_connections]
        n.hidden_connections = [[i, w] for (i, w) in self.hidden_connections]
        n.bias = self.bias

        return n  # type: ignore

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


