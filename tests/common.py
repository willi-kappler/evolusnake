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

    def flip_bit(self):
        pos = rnd.randrange(self.data_size)
        self.data[pos] = 1 - self.data[pos]

    def set_one(self):
        pos = rnd.randrange(self.data_size)
        self.data[pos] = 1

    def set_zero(self):
        pos = rnd.randrange(self.data_size)
        self.data[pos] = 0

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.flip_bit()
            case 1:
                self.set_one()
            case 2:
                self.set_zero()
            case _:
                raise ValueError(f"Unknown mutation operation: {mut_op}")

        self.mutate_called += 1

    @override
    def es_randomize(self):
        for i in range(self.data_size):
            self.data[i] = rnd.randrange(2)

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

        return new  # type: ignore

    @override
    def es_to_json(self) -> dict:
        return {"data": self.data, "fitness": self.fitness}

    @override
    def es_from_json(self, data: dict):
        self.data = data["data"]
        self.data_size = len(self.data)
        self.fitness = data["fitness"]


