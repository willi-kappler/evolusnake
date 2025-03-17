# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the base data class for an individual.
"""

# Python std lib:
import logging
from typing import Any, Self
from sys import float_info

logger = logging.getLogger(__name__)

class ESIndividual:
    def __init__(self):
        self.fitness: float = float_info.max

    def es_mutate(self):
        # Must be implemented by the user
        raise NotImplementedError

    def es_randomize(self):
        # Must be implemented by the user
        raise NotImplementedError

    def es_calculate_fitness(self):
        # Must be implemented by the user
        raise NotImplementedError

    def es_clone(self) -> Self:
        # Must be implemented by the user
        raise NotImplementedError

