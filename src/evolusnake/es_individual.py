# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the base data class for an individual.
"""

# Python std lib:
import logging
from typing import Self
from sys import float_info
from collections import Counter

logger = logging.getLogger(__name__)


class ESIndividual:
    def __init__(self):
        self.fitness: float = float_info.max
        self.mut_op_counter: Counter = Counter()

    def es_reset_counter(self):
        self.mut_op_counter = Counter()

    def es_mutate_internal(self, mut_op: int):
        # Statistics: keep track of how many times
        # each mutation operation has been used.
        self.mut_op_counter[mut_op] += 1
        self.es_mutate(mut_op)

    def es_mutate(self, mut_op: int):
        # Must be implemented by the user.
        raise NotImplementedError

    def es_randomize(self):
        # Must be implemented by the user.
        raise NotImplementedError

    def es_calculate_fitness(self):
        # Must be implemented by the user.
        raise NotImplementedError

    def es_clone_internal(self) -> Self:
        # Clone internal structures.
        clone = self.es_clone()
        clone.mut_op_counter = Counter(self.mut_op_counter)
        clone.fitness = self.fitness
        return clone

    def es_clone(self) -> Self:
        # Must be implemented by the user.
        raise NotImplementedError

    def es_to_json(self) -> dict:
        # Must be implemented by the user.
        raise NotImplementedError

    def es_from_json(self, data: dict):
        # Must be implemented by the user.
        raise NotImplementedError

    def es_actual_fitness(self) -> float:
        # This fitness will be printed.
        # Change it to the actual fitness if
        # needed.
        return self.fitness

    def es_new_best_individual(self):
        # This method is called whenever there is a new best individual
        # on the server (the best of the best). And this individual is the best one.
        pass

