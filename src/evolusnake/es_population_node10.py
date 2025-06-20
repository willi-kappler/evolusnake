# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 10.
"""

# Python std lib:
import logging
from typing import override

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation, ESIterationCallBack

logger = logging.getLogger(__name__)


class ESPopulationNode10(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 10")
        logger.info("Sort population, take the best individual and")
        logger.info("clone and mutate it. Duplicates are allowed.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual, iteration_callback)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode10.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_shuffle_mutation_operations()
        self.population.es_sort_population()
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_fraction_iteration()

            best_ind: ESIndividual = self.population.population[0]

            if best_ind.fitness <= self.population.target_fitness:
                self.population.es_early_exit(i)
                break

            for j in range(1, self.population.population_size):
                new_ind: ESIndividual = best_ind.es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    new_ind.es_mutate_internal(self.population.es_get_mut_op())
                new_ind.es_calculate_fitness()

                self.population.population[j] = new_ind

            self.population.es_sort_population()

        self.population.es_after_iteration()
        self.population.es_calculate_fitness2()
        self.population.es_log_statistics()
        return self.population.es_get_best()

