# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
import pathlib
from typing import override, Self
import random as rnd

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer


logger = logging.getLogger(__name__)


class RosenbrockIndividual(ESIndividual):
    def __init__(self, dimensions: int, lower_bound: float, upper_bound: float):
        super().__init__()

        self.dimensions: int = dimensions
        self.lower_bound: float = lower_bound
        self.upper_bound: float = upper_bound
        self.values: list[float] = []

        for _ in range(self.dimensions):
            self.values.append(self.gen_rnd_value())

    def gen_rnd_value(self) -> float:
        return rnd.uniform(self.lower_bound, self.upper_bound)

    def inc_value(self):
        i = rnd.randrange(self.dimensions)

        self.values[i] += rnd.random()
        if self.values[i] > self.upper_bound:
            self.values[i] = self.lower_bound

    def dec_value(self):
        i = rnd.randrange(self.dimensions)

        self.values[i] -= rnd.random()
        if self.values[i] < self.lower_bound:
            self.values[i] = self.upper_bound

    def swap_values(self):
        i = rnd.randrange(self.dimensions)
        j = rnd.randrange(self.dimensions)

        while i == j:
            j = rnd.randrange(self.dimensions)

        (self.values[i], self.values[j]) = (self.values[j], self.values[i])

    def random_value(self):
        i = rnd.randrange(self.dimensions)

        self.values[i] = self.gen_rnd_value()

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.inc_value()
            case 1:
                self.dec_value()
            case 2:
                self.swap_values()
            case 3:
                self.random_value()
            case _:
                raise ValueError(f"Unknown mutation operation: {mut_op}")

    @override
    def es_randomize(self):
        for i in range(self.dimensions):
            self.values[i] = self.gen_rnd_value()

    @override
    def es_calculate_fitness(self):
        fitness: float = 0.0

        for i in range(self.dimensions - 1):
            term1 = 100.0 * (self.values[i + 1] - self.values[i]**2.0)**2.0
            term2 = (1.0 - self.values[i])**2.0
            fitness += term1 + term2

        self.fitness = fitness

    @override
    def es_clone(self) -> Self:
        new = RosenbrockIndividual(self.dimensions, self.lower_bound, self.upper_bound)
        new.values = self.values
        new.fitness = self.fitness

        return new # type: ignore

    @override
    def es_to_json(self) -> dict:
        data = {
            "dimensions": self.dimensions,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "values": self.values
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.dimensions = data["dimensions"]
        self.lower_bound = data["lower_bound"]
        self.upper_bound = data["upper_bound"]
        self.values = data["values"]

def main():
    config = ESConfiguration.from_json("rosenbrock_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "rosenbrock_server.log"

    if not server_mode:
        rosenbrock_num: int = 1

        while True:
            log_file_name = f"rosenbrock_node_{rosenbrock_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                rosenbrock_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    ind = RosenbrockIndividual(10, -5.0, 5.0)

    # The theoretical minimum is 0.0, when all values are
    # exactly 1.0
    config.target_fitness = 0.0

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
    else:
        print("Create and start node.")
        population = es_select_population(config, ind)
        population.ps_run()

if __name__ == "__main__":
    main()

