# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 2.
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

class ESPopulationNode2(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 2")
        logger.info("Mutate a clone and if it's better keep it.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode2.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_reset_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        minimum_found: bool = False

        for i in range(self.population.num_of_iterations):
            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate()
                tmp_ind.es_calculate_fitness()

                if tmp_ind.fitness < self.population.population[j].fitness:
                    self.population.population[j] = tmp_ind

                    if tmp_ind.fitness <= self.population.target_fitness:
                        logger.info(f"Early exit at iteration {i}")
                        minimum_found = True
                        break

            if minimum_found:
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

        self.population.es_find_best_and_worst_individual()

        best_fitness: float = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")

        return self.population.population[self.population.best_index]

