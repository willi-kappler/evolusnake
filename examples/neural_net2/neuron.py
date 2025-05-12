# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


import random as rnd
from typing import Self
import itertools


class Neuron:
    def __init__(self):
        self.input_connections: list = []
        self.hidden_connections: list = []
        self.current_value: float = 0.0
        self.bias: float = rnd.uniform(-1.0, 1.0)

    def is_empty(self) -> bool:
        return (self.input_connections == []) and (self.hidden_connections == [])

    def mutate_bias(self) -> float:
        prev_value: float = self.bias
        self.bias = rnd.uniform(-1.0, 1.0)
        return self.bias - prev_value

    def has_input_connection(self, new_index):
        for (index2, _) in self.input_connections:
            if new_index == index2:
                return True

        return False

    def add_input_connection(self, new_index: int):
        if self.has_input_connection(new_index):
            self.mutate_input_connection()
            return

        weight: float = rnd.uniform(-1.0, 1.0)
        self.input_connections.append([new_index, weight])

    def remove_input_connection(self):
        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            self.input_connections.pop(index)

    def mutate_input_connection(self) -> tuple[int, float]:
        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.input_connections[index]
            prev_value: float = connection[1]
            connection[1] = rnd.uniform(-1.0, 1.0)
            return (index, connection[1] - prev_value)
        else:
            return (-1, 0.0)

    def replace_input_connection(self, new_index: int):
        if self.has_input_connection(new_index):
            self.mutate_input_connection()
            return

        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.input_connections[index]
            connection[0] = new_index

    def get_random_input_connection(self) -> list:
        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            return self.input_connections[index]
        else:
            return []

    def add_hidden_connection(self, new_index: int):
        for (index2, _) in self.hidden_connections:
            if new_index == index2:
                self.mutate_hidden_connection()
                return

        weight: float = rnd.uniform(-1.0, 1.0)
        self.hidden_connections.append([new_index, weight])

    def remove_hidden_connection(self):
        n: int = len(self.hidden_connections)

        if n > 1:
            index: int = rnd.randrange(n)
            self.hidden_connections.pop(index)

    def mutate_hidden_connection(self) -> tuple[int, float]:
        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.hidden_connections[index]
            prev_value: float = connection[1]
            connection[1] = rnd.uniform(-1.0, 1.0)
            return (index, connection[1] - prev_value)
        else:
            return (-1, 0.0)

    def replace_hidden_connection(self, new_index: int):
        for (index2, _) in self.hidden_connections:
            if new_index == index2:
                self.mutate_hidden_connection()
                return

        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.hidden_connections[index]
            connection[0] = new_index

    def get_random_hidden_connection(self) -> list:
        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            return self.hidden_connections[index]
        else:
            return []

    def apply_mutation(self, mut_op: int, index: int, diff: float):
        if diff > 0.0:
            match mut_op:
                case 0:
                    self.bias = min(1.0, self.bias + 0.001)
                case 1:
                    connection = self.input_connections[index]
                    connection[1] = min(1.0, connection[1] + 0.001)
                case 2:
                    connection = self.hidden_connections[index]
                    connection[1] = min(1.0, connection[1] + 0.001)
        else:
            match mut_op:
                case 0:
                    self.bias = max(-1.0, self.bias - 0.001)
                case 1:
                    connection = self.input_connections[index]
                    connection[1] = max(-1.0, connection[1] - 0.001)
                case 2:
                    connection = self.hidden_connections[index]
                    connection[1] = max(-1.0, connection[1] - 0.001)

    def clear(self):
        self.input_connections = []
        self.hidden_connections = []

    def remove_neuron_connection(self, index: int):
        for (i1, (i2, _)) in enumerate(self.hidden_connections):
            if i2 == index:
                self.hidden_connections.pop(i1)
                return

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

    def abs_weight_sum(self) -> float:
        ws: float = 0.0

        for (_, w) in itertools.chain(self.input_connections, self.hidden_connections):
            ws += abs(w)

        return ws

    def num_of_connections(self) -> int:
        return len(self.input_connections) + len(self.hidden_connections)

