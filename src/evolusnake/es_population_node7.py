# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 1.
"""

# Python std lib:
import logging
from typing import override

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)

class ESPopulationNode7(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 7")
        logger.info("TODO...")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode7.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_random_population()
        self.population.es_sort_population()

        offset: int = int(self.population.population_size / 2)
        previous_best_fitness: float = self.population.population[0].fitness
        previous_best_counter: int = 0
        best_fitness: float = previous_best_fitness
        iter_counter: int = 0

        self.population.es_set_num_iterations()
        logger.debug(f"Iterations: {self.population.num_of_iterations}")

        while True:
            # Create a copy of each individual before mutating it:
            for j in range(offset):
                ind: ESIndividual = self.population.population[j]
                self.population.population[j + offset] = ind.es_clone()

                ind.es_mutate()
                ind.es_calculate_fitness()

            self.population.es_sort_population()

            if self.population.population[0].fitness <= self.population.target_fitness:
                logger.info("Early exit")
                break

            best_fitness = self.population.population[0].fitness

            if previous_best_fitness == best_fitness:
                previous_best_counter += 1
                if previous_best_counter >= self.population.num_of_iterations:
                    break
            else:
                previous_best_fitness = best_fitness
                previous_best_counter = 0

            iter_counter += 1

        self.population.best_fitness = self.population.population[0].fitness
        self.population.worst_fitness = self.population.population[-1].fitness

        best_fitness = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness

        logger.debug(f"{best_fitness=}, {worst_fitness=}")
        logger.debug(f"{iter_counter=}")

        return self.population.es_get_best()

