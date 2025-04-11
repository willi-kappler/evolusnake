# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 8.
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


class ESPopulationNode8(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 8")
        logger.info("Best individual at index 0. Increase factor with index.")
        logger.info("Set limit based on factor and best fitness.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1
        self.limit_factor = 10.0**(1.0 / self.population.population_size)
        logger.debug(f"{self.limit_factor=}")

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode8.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()
        self.population.minimum_found = False

        for i in range(self.population.num_of_iterations):
            for j in range(self.population.population_size):
                ind: ESIndividual = self.population.population[j].es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    ind.es_mutate_internal(self.population.es_get_mut_op())
                ind.es_calculate_fitness()

                current_best_fitness: float = self.population.population[0].fitness
                if ind.fitness < current_best_fitness:
                    self.population.population[0] = ind

                    if ind.fitness <= self.population.target_fitness:
                        self.population.es_early_exit(i)
                        break
                else:
                    if j > 0:
                        fitness_limit: float = current_best_fitness * (self.limit_factor**j)
                        if ind.fitness < fitness_limit:
                            self.population.population[j] = ind
                        elif ind.fitness < self.population.population[j].fitness:
                            self.population.population[j] = ind

            if self.population.minimum_found:
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

        self.population.es_sort_population()
        self.population.es_log_statistics()
        return self.population.es_get_best()

