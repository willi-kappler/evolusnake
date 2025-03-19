# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the Parasnake server class.
"""

# Python std lib:
import logging
import json
import random as rnd
from typing import override, Optional

# External imports:
from parasnake.ps_server import PSServer
from parasnake.ps_nodeid import PSNodeId

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual

logger = logging.getLogger(__name__)

class ESServer(PSServer):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init the evolusnake server.")

        super().__init__(config.parasnake_config)

        self.population: list[ESIndividual] = []
        self.population_size = config.server_population_size

        if self.population_size < 2:
            raise ValueError("Population size must be at least 2.")

        self.target_fitness: float = config.target_fitness
        self.result_filename: str = config.result_filename
        self.save_new_fitness: bool = config.save_new_finess
        self.allow_same_fitness: bool = config.allow_same_fitness
        self.share_only_best: bool = config.share_only_best
        self.new_fitness_counter: int = 0

        for _ in range(self.population_size):
            ind: ESIndividual = individual.es_clone()
            ind.es_mutate()
            ind.es_calculate_fitness()
            self.population.append(ind)

        self.population.sort(key=lambda ind: ind.fitness)

        logger.debug(f"{self.population_size=}, {self.target_fitness=}")
        logger.debug(f"{self.result_filename=}, {self.save_new_fitness=}")
        logger.debug(f"{self.allow_same_fitness=}, {self.share_only_best=}")

    def es_save_data(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.population[0].es_to_json(), f)

    @override
    def ps_is_job_done(self) -> bool:
        return self.population[0].fitness <= self.target_fitness

    @override
    def ps_get_new_data(self, node_id: PSNodeId) -> Optional[ESIndividual]:
        logger.debug(f"Request from node: {node_id}")
        i: int = 0

        if not self.share_only_best:
            # Pick a random individual from the current population of best
            # individuals and return it to the node.
            # (avoid to get stuck in a local minimum)
            i = rnd.randrange(self.population_size)

        return self.population[i]

    @override
    def ps_process_result(self, node_id: PSNodeId, result: ESIndividual):
        logger.debug(f"Got new individual from node: {node_id}")

        new_fitness: float = result.fitness

        if new_fitness < self.population[-1].fitness:
            if not self.allow_same_fitness:
                # Only allow unique individuals:
                for ind in self.population:
                    if ind.fitness == new_fitness:
                        return

            # Overwrite (kill) last (worst) individual:
            self.population[-1] = result

            current_best_fitness: float = self.population[0].fitness

            self.population.sort(key=lambda ind: ind.fitness)

            if new_fitness < current_best_fitness:
                self.new_fitness_counter += 1

                logger.info(f"New best fitness: {new_fitness}, previous: {current_best_fitness}")
                logger.info(f"From node: {node_id}, new fitness counter: {self.new_fitness_counter}")
                logger.debug(f"Worst fitness: {self.population[-1].fitness}")

                if self.save_new_fitness:
                    self.es_save_data(f"{self.new_fitness_counter}_{self.result_filename}")

    @override
    def ps_save_data(self) -> None:
        self.es_save_data(self.result_filename)

