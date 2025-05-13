# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override, Self
import random as rnd

# Local imports:
from dataprovider import DataProvider
from neural_net_base import NeuralNetBase

logger = logging.getLogger(__name__)


class NeuralNetIndividual3(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
            data_provider: DataProvider, network_size: int = 1):
        super().__init__(input_size, output_size, data_provider, network_size)

        self.current_neuron: int = 0

    def mutate_bias1(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        neuron = self.hidden_layer[index]
        neuron.mutate_bias1()

    def mutate_bias2(self):
        index: int = rnd.randrange(self.hidden_layer_size)
        neuron = self.hidden_layer[index]
        neuron.mutate_bias2()

    def mutate_input_connection1(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_input_connection1()

    def mutate_input_connection2(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_input_connection2()

    def mutate_hidden_connection1(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_hidden_connection1()

    def mutate_hidden_connection2(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_hidden_connection2()

    @override
    def description(self) -> str:
        return "NeuralNet3: Optimize one neuron for some time before moving to the next one."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_bias1()
            case 1:
                self.mutate_bias2()
            case 2:
                self.mutate_input_connection1()
            case 3:
                self.mutate_input_connection2()
            case 4:
                self.mutate_hidden_connection1()
            case 5:
                self.mutate_hidden_connection2()
            case 6:
                prob: int = rnd.randrange(1000)
                if prob == 0:
                    self.add_neuron()
                else:
                    self.es_mutate(rnd.randrange(6))
            case 7:
                self.add_input_connection()
            case 8:
                self.add_hidden_connection()
            case 9:
                prob: int = rnd.randrange(1000)
                if prob == 0:
                    self.randomize_all_neurons()
                else:
                    self.es_mutate(rnd.randrange(6))
            case 10:
                prob: int = rnd.randrange(100)
                if prob == 0:
                    self.current_neuron = rnd.randrange(self.hidden_layer_size)
                else:
                    self.es_mutate(rnd.randrange(6))
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 3")
                raise ValueError(f"Unknown operation: {mut_op} in net 3")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual3(self.input_size, self.output_size, self.data_provider)
        clone.hidden_layer = [n.clone() for n in self.hidden_layer]
        clone.hidden_layer_size = self.hidden_layer_size
        clone.current_neuron = self.current_neuron

        return clone  # type: ignore

    @override
    def es_from_server(self, other):
        super().es_from_server(other)
        self.current_neuron = rnd.randrange(self.hidden_layer_size)

    @override
    def es_from_json(self, data: dict):
        super().es_from_json(data)
        self.current_neuron = rnd.randrange(self.hidden_layer_size)
