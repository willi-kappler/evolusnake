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
from evolusnake.es_population_node1 import ESPopulationNode1

from evolusnake.es_server import ESServer


logger = logging.getLogger(__name__)


class TSPIndividual(ESIndividual):
    def __init__(self):
        super().__init__()

        self.data: list = []

    @override
    def es_mutate(self):
        pass

    @override
    def es_randomize(self):
        pass

    @override
    def es_calculate_fitness(self):
        pass

    @override
    def es_clone(self) -> Self:
        new = TSPIndividual()
        return new # type: ignore

    @override
    def es_to_json(self) -> dict:
        return {}

    @override
    def es_from_json(self, data: dict):
        pass




# type
#     TSPIndividual* = ref object of NAIndividual
#         data: seq[(float64, float64)]
#
# proc naCalculateFitness2(self: var TSPIndividual): float64 =
#     var length: float64 = 0.0
#     let last = self.data.high
#
#     for i in 1..<last:
#         let dx = self.data[i - 1][0] - self.data[i][0]
#         let dy = self.data[i - 1][1] - self.data[i][1]
#         let d = hypot(dx, dy)
#         length += d
#
#     let dx = self.data[0][0] - self.data[last][0]
#     let dy = self.data[0][1] - self.data[last][1]
#     let d = hypot(dx, dy)
#     length += d
#
#     return length
#
# method naMutate*(self: var TSPIndividual) =
#     let last = self.data.high
#     var i = rand(last)
#     var j = rand(last)
#
#     # Ensure that i != j
#     while i == j:
#         j = rand(last)
#
#     if i > j:
#         swap(i, j)
#
#     # Reverse order
#
#     let d = j - i
#
#     for k in 0..<d:
#         let u = i+k
#         let v = j-k
#         if u >= v:
#             break
#         swap(self.data[u], self.data[v])
#
# method naRandomize*(self: var TSPIndividual) =
#     shuffle(self.data)
#
# method naCalculateFitness*(self: var TSPIndividual) =
#     self.fitness = self.naCalculateFitness2()
#
# method naClone*(self: TSPIndividual): NAIndividual =
#     result = TSPIndividual(data: self.data)
#     result.fitness = self.fitness
#




def main():
    config = ESConfiguration.from_json("tsp_config.json")

    server_mode = False

    log_file_name: str = "tsp_server.log"

    if not server_mode:
        tsp_num: int = 1

        while True:
            log_file_name = f"tsp_node_{tsp_num}.log"
            p = pathlib.Path(log_file_name)

            if p.is_file():
                tsp_num += 1
            else:
                break



    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    ind = TSPIndividual()

    if server_mode:
        server = ESServer(config, ind)
        server.ps_run()
    else:
        population = ESPopulationNode1(config, ind)
        population.ps_run()

if __name__ == "__main__":
    main()

