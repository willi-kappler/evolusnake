# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 6.
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

class ESPopulationNode6(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 6")
        logger.info("TODO...")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode6.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        minimum_found: bool = False

        for i in range(self.population.num_of_iterations):
            for j in range(self.population.population_size):
                tmp_ind1: ESIndividual = self.population.population[j].es_clone()
                best_ind: ESIndividual = tmp_ind1.es_clone()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind1.es_mutate()
                    tmp_ind1.es_calculate_fitness()
                    if tmp_ind1.fitness < best_ind.fitness:
                        best_ind = tmp_ind1.es_clone()

                    tmp_ind2: ESIndividual = self.population.population[j].es_clone()
                    tmp_ind2.es_mutate()
                    tmp_ind2.es_calculate_fitness()
                    if tmp_ind2.fitness < best_ind.fitness:
                        best_ind = tmp_ind2.es_clone()

                if best_ind.fitness < self.population.population[j].fitness:
                    self.population.population[j] = best_ind

                    if best_ind.fitness <= self.population.target_fitness:
                        logger.info(f"Early exit at iteration {i}")
                        minimum_found = True
                        break

            if minimum_found:
                break

        self.population.es_find_best_and_worst_individual()

        best_fitness: float = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")

        return self.population.es_get_best()

