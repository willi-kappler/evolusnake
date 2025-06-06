# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
from typing import override, Self

# Local imports:
import evolusnake.es_utils as utils

from dataprovider import DataProvider
from neural_net_base import NeuralNetBase

logger = logging.getLogger(__name__)


class NeuralNetIndividual7(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
                 data_provider: DataProvider, use_softmax: bool = False,
                 max_size: int = 100):
        super().__init__(input_size, output_size, data_provider, use_softmax, max_size)

        self.prob: int = 100

    @override
    def description(self) -> str:
        return "NeuralNet7: Randomly change values, grow network and connections slowly. " \
            "Use more time to optimize current shape."

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
                if utils.es_rand_int(self.prob) == 0:
                    self.add_input_connection()
                else:
                    self.mutate_hidden_connection2()
            case 7:
                if utils.es_rand_int(self.prob) == 0:
                    self.add_hidden_connection()
                else:
                    self.mutate_hidden_connection2()
            case 8:
                if utils.es_rand_int(self.prob) == 0:
                    self.add_neuron()
                else:
                    self.mutate_hidden_connection2()
            case 9:
                if utils.es_rand_int(self.prob) == 0:
                    self.change_activation_function()
                else:
                    self.mutate_hidden_connection2()
            case _:
                logger.error(f"Unknown operation: {mut_op} in net 1")
                raise ValueError(f"Unknown operation: {mut_op} in net 1")

    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual7(self.input_size, self.output_size, self.data_provider,
                    self.use_softmax, self.max_size)
        return self.clone_base(clone)  # type: ignore
