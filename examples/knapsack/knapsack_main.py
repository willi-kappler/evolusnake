# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
import pathlib
from typing import override, Self

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer
import evolusnake.es_utils as utils


logger = logging.getLogger(__name__)


class KnapsackIndividual(ESIndividual):
    def __init__(self, items: list, capacity: float):
        super().__init__()

        self.items: list = items
        self.capacity: float = capacity
        self.num_items: int = len(items)
        self.selection: list[int] = []

        self.es_randomize()
        self.reset_penalty()

    def reset_penalty(self):
        self.penalty: float = 0.0

        for (value, _) in self.items:
            self.penalty += value

    def flip(self):
        i: int = utils.es_rand_int(self.num_items)
        self.selection[i] = 1 - self.selection[i]

    def to_one(self):
        i: int = utils.es_rand_int(self.num_items)
        self.selection[i] = 1

    def to_zero(self):
        i: int = utils.es_rand_int(self.num_items)
        self.selection[i] = 0

    def swap(self):
        utils.es_random_swap(self.selection)

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.flip()
            case 1:
                self.to_one()
            case 2:
                self.to_zero()
            case 3:
                self.swap()
            case _:
                raise ValueError(f"Unknown mutation operation: {mut_op}")

    @override
    def es_randomize(self):
        for _ in range(self.num_items):
            b: int = utils.es_rand_int(2)
            self.selection.append(b)

    @override
    def es_calculate_fitness(self):
        total_value: float = 0.0
        total_weight: float = 0.0

        for i in range(self.num_items):
            if self.selection[i] == 1:
                (value, weight) = self.items[i]
                total_value += value
                total_weight += weight

                if total_weight > self.capacity:
                    self.fitness = self.penalty
                    return

        self.fitness = self.penalty - total_value

    @override
    def es_clone(self) -> Self:
        new = KnapsackIndividual(self.items, self.capacity)
        new.selection = self.selection[:]

        return new  # type: ignore

    @override
    def es_from_server(self, other):
        self.selection = other.selection

    @override
    def es_to_json(self) -> dict:
        data = {
            "items": self.items,
            "capacity": self.capacity,
            "selection": self.selection
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.items = data["items"]
        self.num_items = len(data)
        self.capacity = data["capacity"]
        self.selection = data["selection"]
        self.reset_penalty()

    @override
    def es_actual_fitness(self) -> float:
        return self.penalty - self.fitness


def main():
    config = ESConfiguration.from_json("knapsack_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "knapsack_server.log"

    if not server_mode:
        knapsack_num: int = 1

        while True:
            log_file_name = f"knapsack_node_{knapsack_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                knapsack_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    ind = KnapsackIndividual([
        (27.98, 33.36),
        (25.27, 24.44),
        (31.32, 35.00),
        (29.84, 20.32),
        (21.34, 26.59),
        (30.18, 25.71),
        (39.13, 24.73),
        (20.38, 23.90),
        (24.44, 21.37),
        (25.01, 18.32),
        (10.83, 15.77),
        (19.99, 13.25),
        (16.54, 17.90),
        (39.87, 35.21)
    ], 100.0)

    config.target_fitness = 224.0

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

