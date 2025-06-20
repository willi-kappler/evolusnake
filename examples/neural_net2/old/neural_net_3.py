# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
from typing import override, Self

# Local imports:
import evolusnake.es_utils as utils

from dataprovider import DataProvider
from neural_net_base import NeuralNetBase

logger = logging.getLogger(__name__)


class NeuralNetIndividual3(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, use_softmax: bool = False, max_size: int = 0):
        super().__init__(input_size, output_size, data_provider, use_softmax, max_size)

        self.current_neuron: int = 0

    def mutate_current_bias1(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_bias1()

    def mutate_current_bias2(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_bias2()

    def mutate_current_input_connection1(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_input_connection1()

    def mutate_current_input_connection2(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_input_connection2()

    def mutate_current_hidden_connection1(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_hidden_connection1()

    def mutate_current_hidden_connection2(self):
        neuron = self.hidden_layer[self.current_neuron]
        neuron.mutate_hidden_connection2()

    @override
    def description(self) -> str:
        return "NeuralNet3: Optimize one neuron for some time before moving to the next one."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.mutate_current_bias1()
            case 1:
                self.mutate_current_bias2()
            case 2:
                self.mutate_current_input_connection1()
            case 3:
                self.mutate_current_input_connection2()
            case 4:
                self.mutate_current_hidden_connection1()
            case 5:
                self.mutate_current_hidden_connection2()
            case 6:
                # Hyperparameter: 100
                prob: int = utils.es_rand_int(100)
                if prob == 0:
                    self.current_neuron = utils.es_rand_int(self.hidden_layer_size)
                else:
                    self.es_mutate(utils.es_rand_int(6))
            case 7:
                if self.common_mutations():
                    self.es_mutate(utils.es_rand_int(6))
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 3")
                raise ValueError(f"Unknown operation: {mut_op} in net 3")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual3(self.input_size, self.output_size, self.data_provider,
                    self.use_softmax, self.max_size)
        clone.current_neuron = self.current_neuron
        return self.clone_base(clone)  # type: ignore

    @override
    def es_from_server(self, other):
        super().es_from_server(other)
        self.current_neuron = utils.es_rand_int(self.hidden_layer_size)

    @override
    def es_from_json(self, data: dict):
        super().es_from_json(data)
        self.current_neuron = utils.es_rand_int(self.hidden_layer_size)
