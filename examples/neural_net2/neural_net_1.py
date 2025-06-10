# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
from typing import override, Self

# Local imports:
# import evolusnake.es_utils as utils

from dataprovider import DataProvider
from neural_net_base import NeuralNetBase
from neuron import Neuron

logger = logging.getLogger(__name__)


class NeuralNetIndividual1(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, use_softmax: bool = False, max_size: int = 100):
        super().__init__(input_size, output_size, data_provider, use_softmax, max_size)

    @override
    def add_neuron(self):
        if self.hidden_layer_size < self.max_size:
            new_neuron: Neuron = Neuron()

            # Add connections to all inputs:
            for i in range(self.input_size):
                new_neuron.add_input_connection(i)

            # Add connections to all other neurons:
            for i in range(self.hidden_layer_size):
                new_neuron.add_hidden_connection(i)

            # Add a connection from all other neurons to the new one:
            for i in range(self.hidden_layer_size):
                self.hidden_layer[i].add_hidden_connection(self.hidden_layer_size)

            self.hidden_layer.append(new_neuron)
            self.hidden_layer_size += 1
        else:
            self.mutate_hidden_connection2()

    @override
    def description(self) -> str:
        return "NeuralNet1: Use fully connected neurons."

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
                self.add_input_connection()
            case 7:
                self.add_hidden_connection()
            case 8:
                self.add_neuron()
            case 9:
                self.change_activation_function()
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 1")
                raise ValueError(f"Unknown operation: {mut_op} in net 1")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual1(self.input_size, self.output_size, self.data_provider,
                    self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore
