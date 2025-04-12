# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 9.
"""

# Python std lib:
import logging
from typing import override
import random as rnd

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)


class ESPopulationNode9(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 9")
        logger.info("")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode9.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()

        for i in range(self.population.num_of_iterations):
            self.population.es_sort_population()
            single_ind: ESIndividual = self.population.population[0]

            if single_ind.fitness <= self.population.target_fitness:
                self.population.es_early_exit(i)
                break

            self.population.population = [single_ind]
            current_size: int = 1

            new_ind: ESIndividual = single_ind.es_clone_internal()

            while current_size < self.population.population_size:
                new_ind.es_mutate_internal(self.population.es_get_mut_op())
                new_ind.es_calculate_fitness()

                already_in_population: bool = False

                for ind in self.population.population:
                    if new_ind.fitness == ind.fitness:
                        already_in_population = True
                        break

                if not already_in_population:
                    self.population.population.append(new_ind.es_clone_internal())
                    current_size += 1

        self.population.es_sort_population()
        self.population.es_log_statistics()
        return self.population.es_get_best()

