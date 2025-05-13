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



class NeuralNetIndividual2(NeuralNetBase):
    def __init__(self, input_size: int, output_size: int,
            data_provider: DataProvider, network_size: int = 1):
        super().__init__(input_size, output_size, data_provider, network_size)

    @override
    def description(self) -> str:
        return "NeuralNet2: Try two different small mutations and compare them to the current fitness. Use the best one."

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                pass


    @override
    def es_clone(self) -> Self:
        clone = NeuralNetIndividual2(self.input_size, self.output_size, self.data_provider)
        clone.hidden_layer = [n.clone() for n in self.hidden_layer]
        clone.hidden_layer_size = self.hidden_layer_size

        return clone  # type: ignore

