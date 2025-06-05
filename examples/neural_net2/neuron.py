# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
from typing import Self
import itertools
import math

# Local imports:
import evolusnake.es_utils as utils


class Neuron:
    def __init__(self):
        self.input_connections: list = []
        self.input_connections_size: int = 0
        self.hidden_connections: list = []
        self.hidden_connections_size: int = 0
        self.current_value: float = 0.0
        self.bias: float = utils.es_uniform1()
        self.bias_delta: float = 0.0
        self.activation_kind: int = 0

    def is_empty(self) -> bool:
        return (self.input_connections_size == 0) and (self.hidden_connections_size == 0)

    def mutate_bias1(self):
        self.bias = utils.es_uniform1()

    def mutate_bias2(self):
        self.bias += utils.es_uniform2()
        self.bias = min(1.0, max(-1.0, self.bias))

    def mutate_bias3(self):
        self.bias += self.bias_delta
        if self.bias < -1.0:
            self.bias = -1.0
            self.bias_delta = -self.bias_delta
        elif self.bias > 1.0:
            self.bias = 1.0
            self.bias_delta = -self.bias_delta

    def change_bias_delta(self):
        self.bias_delta: float = utils.es_uniform2()

    def has_input_connection(self, new_index):
        for (index2, _, _) in self.input_connections:
            if new_index == index2:
                return True

        return False

    def add_input_connection(self, new_index: int):
        if self.has_input_connection(new_index):
            self.mutate_input_connection2()
            return

        weight: float = utils.es_uniform1()
        delta: float = 0.0
        self.input_connections.append([new_index, weight, delta])
        self.input_connections_size += 1

    def mutate_input_connection1(self):
        if self.input_connections_size > 0:
            index: int = utils.es_rand_int(self.input_connections_size)
            connection: list = self.input_connections[index]
            connection[1] = utils.es_uniform1()

    def mutate_input_connection2(self):
        if self.input_connections_size > 0:
            index: int = utils.es_rand_int(self.input_connections_size)
            connection: list = self.input_connections[index]
            connection[1] += utils.es_uniform2()
            connection[1] = min(1.0, max(-1.0, connection[1]))

    def mutate_input_connection3(self):
        if self.input_connections_size > 0:
            index: int = utils.es_rand_int(self.input_connections_size)
            connection: list = self.input_connections[index]
            connection[1] += connection[2]  # Add delta
            v: float = connection[1]
            d: float = connection[2]
            if v < -1.0:
                connection[1] = -1.0
                connection[2] = -d
            elif v > 1.0:
                connection[1] = 1.0
                connection[2] = -d

    def get_random_input_connection(self) -> list:
        if self.input_connections_size > 0:
            index: int = utils.es_rand_int(self.input_connections_size)
            return self.input_connections[index]
        else:
            return []

    def has_hidden_connection(self, new_index: int):
        for (index2, _, _) in self.hidden_connections:
            if new_index == index2:
                return True

        return False

    def add_hidden_connection(self, new_index: int):
        if self.has_hidden_connection(new_index):
            self.mutate_hidden_connection2()
            return

        weight: float = utils.es_uniform1()
        delta: float = 0.0
        self.hidden_connections.append([new_index, weight, delta])
        self.hidden_connections_size += 1

    def mutate_hidden_connection1(self):
        if self.hidden_connections_size > 0:
            index: int = utils.es_rand_int(self.hidden_connections_size)
            connection: list = self.hidden_connections[index]
            connection[1] = utils.es_uniform1()

    def mutate_hidden_connection2(self):
        if self.hidden_connections_size > 0:
            index: int = utils.es_rand_int(self.hidden_connections_size)
            connection: list = self.hidden_connections[index]
            connection[1] += utils.es_uniform2()
            connection[1] = min(1.0, max(-1.0, connection[1]))

    def mutate_hidden_connection3(self):
        if self.hidden_connections_size > 0:
            index: int = utils.es_rand_int(self.hidden_connections_size)
            connection: list = self.hidden_connections[index]
            connection[1] += connection[2]  # Add delta
            v: float = connection[1]
            d: float = connection[2]
            if v < -1.0:
                connection[1] = -1.0
                connection[2] = -d
            elif v > 1.0:
                connection[1] = 1.0
                connection[2] = -d

    def get_random_hidden_connection(self) -> list:
        if self.hidden_connections_size > 0:
            index: int = utils.es_rand_int(self.hidden_connections_size)
            return self.hidden_connections[index]
        else:
            return []

    def change_all_deltas(self):
        self.change_bias_delta()

        for connection in itertools.chain(self.input_connections, self.hidden_connections):
            connection[2] = utils.es_uniform2()

    def mutate_all_values(self):
        self.mutate_bias3()

        for connection in itertools.chain(self.input_connections, self.hidden_connections):
            connection[1] += connection[2]
            connection[1] = min(1.0, max(-1.0, connection[1]))

    def mutate_activation(self):
        self.activation_kind = utils.es_rand_int(5)

    def randomize_all_values(self):
        self.mutate_bias1()

        for connection in itertools.chain(self.input_connections, self.hidden_connections):
            connection[1] = utils.es_uniform1()
            connection[2] = utils.es_uniform2()

    def remove_input_connection(self):
        if self.input_connections_size > 0:
            index: int = utils.es_rand_int(self.input_connections_size)
            self.input_connections.pop(index)
            self.input_connections_size -= 1

    def remove_hidden_connection(self):
        if self.hidden_connections_size > 0:
            index: int = utils.es_rand_int(self.hidden_connections_size)
            self.hidden_connections.pop(index)
            self.hidden_connections_size -= 1

    def remove_connection_to(self, index):
        for i in range(self.hidden_connections_size):
            if self.hidden_connections[i][0] == index:
                self.hidden_connections.pop(i)
                self.hidden_connections_size -= 1
                break

    def remove_all_connections(self):
        self.input_connections = []
        self.input_connections_size = 0
        self.hidden_connections = []
        self.hidden_connections_size = 0

    def prune_connections(self):
        # Hyperparameter: 0.01
        self.input_connections = [connection for connection in self.input_connections if abs(connection[1]) > 0.01]
        self.input_connections_size = len(self.input_connections)

        # Hyperparameter: 0.01
        self.hidden_connections = [connection for connection in self.hidden_connections if abs(connection[1]) > 0.01]
        self.hidden_connections_size = len(self.hidden_connections)

    def split_neuron(self) -> "Neuron":
        new_neuron: Neuron = Neuron()

        new_neuron.bias = self.bias
        new_neuron.activation_kind = self.activation_kind

        new_input_connections = []
        old_input_connections = []

        half_size: int = self.input_connections_size // 2

        for i in range(half_size):
            old_input_connections.append(self.input_connections[i])

        for i in range(half_size, self.input_connections_size):
            new_input_connections.append(self.input_connections[i])

        self.input_connections = old_input_connections
        self.input_connections_size = len(old_input_connections)

        new_neuron.input_connections = new_input_connections
        new_neuron.input_connections_size = len(new_input_connections)

        new_hidden_connections = []
        old_hidden_connections = []

        half_size: int = self.hidden_connections_size // 2

        for i in range(half_size):
            old_hidden_connections.append(self.hidden_connections[i])

        for i in range(half_size, self.hidden_connections_size):
            new_hidden_connections.append(self.hidden_connections[i])

        self.hidden_connections = old_hidden_connections
        self.hidden_connections_size = len(old_hidden_connections)

        new_neuron.hidden_connections = new_hidden_connections
        new_neuron.hidden_connections_size = len(new_hidden_connections)

        return new_neuron

    def evaluate(self, input_values: list, hidden_layer: list):
        new_value: float = self.bias

        for (index, weight, _) in self.input_connections:
            new_value += weight * input_values[index]

        for (index, weight, _) in self.hidden_connections:
            new_value += weight * hidden_layer[index].current_value

        # Limit value to avoid overflow in exp() function below:
        new_value = max(-20.0, min(20.0, new_value))

        match self.activation_kind:
            case 0:
                # ReLU
                self.current_value = max(0, new_value)
            case 1:
                # Sigmoid
                self.current_value = 1 / (1 + math.exp(-new_value))
            case 2:
                # Hyperbolic tangent
                self.current_value = ((math.exp(new_value) - math.exp(-new_value)) /
                    (math.exp(new_value) + math.exp(-new_value)))
            case 3:
                # Leaky ReLU
                self.current_value = new_value if new_value >= 0 else 0.01 * new_value
            case 4:
                # Exponential linear unit (ELU)
                self.current_value = new_value if new_value >= 0 else 1.0 * (math.exp(new_value) - 1)
            case _:
                raise ValueError(f"Unknown activation function: {self.activation_kind}")

    def clone(self) -> Self:
        n = Neuron()
        # Do proper cloning!
        n.input_connections = [[i, w, d] for (i, w, d) in self.input_connections]
        n.input_connections_size = self.input_connections_size
        n.hidden_connections = [[i, w, d] for (i, w, d) in self.hidden_connections]
        n.hidden_connections_size = self.hidden_connections_size
        n.bias = self.bias
        n.bias_delta = self.bias_delta
        n.activation_kind = self.activation_kind

        return n  # type: ignore

    def to_json(self) -> dict:
        data = {
            "input_connections": self.input_connections,
            "hidden_connections": self.hidden_connections,
            "bias": self.bias,
            "activation_kind": self.activation_kind
        }

        return data

    def from_json(self, data: dict):
        self.input_connections = data["input_connections"]
        self.input_connections_size = len(self.input_connections)
        self.hidden_connections = data["hidden_connections"]
        self.hidden_connections_size = len(self.hidden_connections)
        self.bias = data["bias"]
        self.activation_kind = data["activation_kind"]

    def abs_weight_sum(self) -> float:
        ws: float = 0.0

        for (_, w, _) in itertools.chain(self.input_connections, self.hidden_connections):
            ws += abs(w)

        return ws

    def num_of_connections(self) -> int:
        return self.input_connections_size + self.hidden_connections_size
