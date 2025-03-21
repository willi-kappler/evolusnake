# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
import pathlib
from typing import override, Self
import random as rnd
import math

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer


logger = logging.getLogger(__name__)


class TSPIndividual(ESIndividual):
    def __init__(self):
        super().__init__()

        self.positions: list = []
        self.num_elems: int = 0

    def load_data(self, filename):
        self.positions = []
        num_elems = 0

        with open(filename, "r") as f:
            for line in f:
                items = line.split(" ")
                x = float(items[0])
                y = float(items[1])

                self.positions.append((x, y))
                num_elems += 1

        self.num_elems = num_elems

    @override
    def es_mutate(self):
        i1: int = rnd.randrange(self.num_elems)
        i2: int = rnd.randrange(self.num_elems)

        while i1 == i2:
            i2 = rnd.randrange(self.num_elems)

        if i1 > i2:
            (i1, i2) = (i2, i1)

        while i1 < i2:
            (self.positions[i1], self.positions[i2]) = (self.positions[i2], self.positions[i1])

            i1 += 1
            i2 -= 1

    @override
    def es_randomize(self):
        rnd.shuffle(self.positions)

    @override
    def es_calculate_fitness(self):
        length = 0.0

        x0: float
        y0: float
        x1: float
        y1: float

        (x0, y0) = self.positions[-1]

        for (x1, y1) in self.positions:
            length += math.hypot(x0 - x1, y0 - y1)
            (x0, y0) = (x1, y1)

        self.fitness = length

    @override
    def es_clone(self) -> Self:
        new = TSPIndividual()
        new.positions = self.positions[:]
        new.num_elems = self.num_elems
        new.fitness = self.fitness

        return new # type: ignore

    @override
    def es_to_json(self) -> dict:
        data = {
            "fitness": self.fitness,
            "positions": self.positions
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.fitness = data["fitness"]
        self.positions = data["positions"]
        self.num_elems = len(self.positions)


def main():
    config = ESConfiguration.from_json("tsp_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "tsp_server.log"

    if not server_mode:
        tsp_num: int = 1

        while True:
            log_file_name = f"tsp_node_{tsp_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                tsp_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    ind = TSPIndividual()

    # Best fitness with city_positions1: 376.3341189874508
    # Possible good limit: 380.0
    # ind.load_data("city_positions1.txt")
    # config.target_fitness = 380.0

    # Best fitness with city_positions2: 8219.322949926998
    # Possible good limit: 8230.0
    ind.load_data("city_positions2.txt")
    config.target_fitness = 8230.0
    #config.target_fitness = 9000.0

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

