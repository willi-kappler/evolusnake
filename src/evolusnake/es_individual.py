# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the base data class for an individual.
"""

# Python std lib:
from typing import Any

class ESIndividual:
    def __init__(self):
        self.fitness = 0.0

    def es_mutate(self):
        pass

    def es_randomize(self):
        pass

    def es_calculate_fitness(self):
        pass

    def __lt__(self, other: Any) -> bool:
        match other:
            case float(f):
                return self.fitness < f
            case ESIndividual() as ind:
                return self.fitness < ind.fitness
            case _:
                raise ValueError(f"Type not supported: {other}")

    def __le__(self, other: Any) -> bool:
        match other:
            case float(f):
                return self.fitness <= f
            case ESIndividual() as ind:
                return self.fitness <= ind.fitness
            case _:
                raise ValueError(f"Type not supported: {other}")


