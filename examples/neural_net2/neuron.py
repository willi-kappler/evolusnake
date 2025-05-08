# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


import random as rnd
from typing import Self


class Neuron:
    def __init__(self):
        self.input_connections: list = []
        self.hidden_connections: list = []
        self.current_value: float = 0.0
        self.delta_limit1: float = 0.9
        self.delta_limit2: float = self.delta_limit1 * 0.01
        self.bias: float = rnd.uniform(-self.delta_limit1, self.delta_limit1)

    def is_empty(self) -> bool:
        return (self.input_connections == []) and (self.hidden_connections == [])

    def change_delta(self, value: float) -> float:
        delta: float = rnd.uniform(-self.delta_limit2, self.delta_limit2)
        value += delta
        if value < -self.delta_limit1:
            return -self.delta_limit1
        elif value > self.delta_limit1:
            return self.delta_limit1
        else:
            return value

    def mutate_bias(self):
        self.bias = self.change_delta(self.bias)

    def add_input_connection(self, index: int):
        for (index2, _) in self.input_connections:
            if index == index2:
                return

        weight: float = rnd.uniform(-self.delta_limit1, self.delta_limit1)
        self.input_connections.append([index, weight])

    def mutate_input_connection(self):
        l: int = len(self.input_connections)

        if l > 0:
            index: int = rnd.randrange(l)
            connection: list = self.input_connections[index]
            connection[1] = self.change_delta(connection[1])

    def replace_input_connection(self, new_index: int):
        l: int = len(self.input_connections)

        if l > 0:
            index: int = rnd.randrange(l)
            connection: list = self.input_connections[index]
            connection[0] = new_index

    def add_hidden_connection(self, index: int):
        for (index2, _) in self.hidden_connections:
            if index == index2:
                return

        weight: float = rnd.uniform(-self.delta_limit1, self.delta_limit1)
        self.hidden_connections.append([index, weight])

    def mutate_hidden_connection(self):
        l: int = len(self.hidden_connections)

        if l > 0:
            index: int = rnd.randrange(l)
            connection: list = self.hidden_connections[index]
            connection[1] = self.change_delta(connection[1])

    def replace_hidden_connection(self, new_index: int):
        l: int = len(self.hidden_connections)

        if l > 0:
            index: int = rnd.randrange(l)
            connection: list = self.hidden_connections[index]
            connection[0] = new_index

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

    def biggest_weight(self) -> float:
        bw = 0.0

        for (_, w) in self.input_connections:
            bw = max(abs(w), bw)

        for (_, w) in self.hidden_connections:
            bw = max(abs(w), bw)

        return bw

