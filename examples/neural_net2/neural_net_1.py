# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
from typing import override, Self

# External libraries:
import fastrand

# Local imports:
from dataprovider import DataProvider
from neural_net_base import NeuralNetBase
from neuron import Neuron

logger = logging.getLogger(__name__)


class NeuralNetIndividual1(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, network_size: int = 0,
                 use_softmax: bool = False, max_size: int = 100):
        super().__init__(input_size, output_size, data_provider, network_size,
                 use_softmax, max_size)

    def mutate_bias1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_bias1()

    def mutate_bias2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_bias2()

    def mutate_input_connection1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_input_connection1()

    def mutate_input_connection2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_input_connection2()

    def mutate_hidden_connection1(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_hidden_connection1()

    def mutate_hidden_connection2(self):
        neuron: Neuron = self.get_random_neuron()[0]
        neuron.mutate_hidden_connection2()

    @override
    def description(self) -> str:
        return "NeuralNet1: Just randomly change values."

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
                if self.common_mutations():
                    self.es_mutate(fastrand.pcg32bounded(6))
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 1")
                raise ValueError(f"Unknown operation: {mut_op} in net 1")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual1(self.input_size, self.output_size, self.data_provider,
                    self.network_size, self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore
