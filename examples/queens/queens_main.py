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


class QueensIndividual(ESIndividual):
    def __init__(self, size: int):
        super().__init__()

        self.positions: list = []
        self.num_elems: int = size

        for i in range(size):
            self.positions.append([i, 0])

    def random_pos(self):
        i: int = rnd.randrange(self.num_elems)
        y: int = rnd.randrange(self.num_elems)

        self.positions[i][1] = y

    def swap_pos(self):
        i: int = rnd.randrange(self.num_elems)
        j: int = rnd.randrange(self.num_elems)

        while i == j:
            j = rnd.randrange(self.num_elems)

        (self.positions[i][1], self.positions[j][1]) = (self.positions[j][1], self.positions[i][1])

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.random_pos()
            case 1:
                self.swap_pos()
            case _:
                raise ValueError(f"Unknown mutation operation: {mut_op}")

    @override
    def es_randomize(self):
        for i in range(self.num_elems):
            y: int = rnd.randrange(self.num_elems)
            self.positions[i][1] = y

    @override
    def es_calculate_fitness(self):
        error: int = 0
        x1: int
        y1: int
        x2: int
        y2: int

        for i in range(self.num_elems):
            (x1, y1) = self.positions[i]
            for j in range(i+1, self.num_elems):
                (x2, y2) = self.positions[j]
                if y1 == y2:
                    error += 1

                dx = abs(x1 - x2)
                dy = abs(y1 - y2)

                if dx == dy:
                    error += 1

        self.fitness = float(error)

    @override
    def es_clone(self) -> Self:
        new = QueensIndividual(self.num_elems)
        new.positions = self.positions[:]
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
    config = ESConfiguration.from_json("queens_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "queens_server.log"

    if not server_mode:
        queens_num: int = 1

        while True:
            log_file_name = f"queens_node_{queens_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                queens_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    ind = QueensIndividual(10)

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

