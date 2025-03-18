# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
from typing import override, Self
import random as rnd

# Local imports:
from evolusnake.es_individual import ESIndividual

class TestIndividual(ESIndividual):
    def __init__(self):
        super().__init__()

        self.mutate_called: int = 0
        self.calc_fitness_called: int = 0
        self.clone_called: int = 0
        self.randomize_called: int = 0

        self.data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.data_size = 10

    @override
    def es_mutate(self):
        pos = rnd.randrange(self.data_size)
        self.data[pos] = 1 - self.data[pos]
        self.mutate_called += 1

    @override
    def es_randomize(self):
        max_random = self.data_size * 4
        for _ in range(max_random):
            self.es_mutate()

        self.randomize_called += 1

    @override
    def es_calculate_fitness(self):
        self.fitness = float(sum(self.data))
        self.calc_fitness_called += 1

    @override
    def es_clone(self) -> Self:
        new: TestIndividual = TestIndividual()
        new.data = self.data[:]
        new.data_size = self.data_size
        new.fitness = self.fitness

        self.clone_called += 1

        return new # type: ignore

    @override
    def es_to_json(self) -> dict:
        return {"data": self.data, "fitness": self.fitness}

    @override
    def es_from_json(self, data: dict):
        self.data = data["data"]
        self.data_size = len(self.data)
        self.fitness = data["fitness"]


