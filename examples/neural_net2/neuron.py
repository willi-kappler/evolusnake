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

    def mutate_bias1(self):
        self.bias = rnd.uniform(-1.0, 1.0)

    def mutate_bias2(self):
        self.bias += rnd.uniform(-0.01, 0.01)
        self.bias = min(1.0, max(-1.0, self.bias))

    def has_input_connection(self, new_index):
        for (index2, _) in self.input_connections:
            if new_index == index2:
                return True

        return False

    def add_input_connection(self, new_index: int):
        if self.has_input_connection(new_index):
            self.mutate_input_connection1()
            return

        weight: float = rnd.uniform(-1.0, 1.0)
        self.input_connections.append([new_index, weight])

    def mutate_input_connection1(self):
        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.input_connections[index]
            connection[1] = rnd.uniform(-1.0, 1.0)

    def mutate_input_connection2(self):
        n: int = len(self.input_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.input_connections[index]
            connection[1] += rnd.uniform(-0.01, 0.01)
            connection[1] = min(1.0, max(-1.0, connection[1]))

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
                self.mutate_hidden_connection1()
                return

        weight: float = rnd.uniform(-1.0, 1.0)
        self.hidden_connections.append([new_index, weight])

    def mutate_hidden_connection1(self):
        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.hidden_connections[index]
            connection[1] = rnd.uniform(-1.0, 1.0)

    def mutate_hidden_connection2(self):
        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            connection: list = self.hidden_connections[index]
            connection[1] += rnd.uniform(-0.01, 0.01)
            connection[1] = min(1.0, max(-1.0, connection[1]))

    def get_random_hidden_connection(self) -> list:
        n: int = len(self.hidden_connections)

        if n > 0:
            index: int = rnd.randrange(n)
            return self.hidden_connections[index]
        else:
            return []

    def randomize_all_values(self):
        self.mutate_bias1()

        for connection in itertools.chain(self.input_connections, self.hidden_connections):
            connection[1] = rnd.uniform(-1.0, 1.0)

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

