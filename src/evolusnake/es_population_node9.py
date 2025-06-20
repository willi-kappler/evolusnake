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

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation, ESIterationCallBack

logger = logging.getLogger(__name__)


class ESPopulationNode9(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 9")
        logger.info("Sort population and take the best individual.")
        logger.info("Repopulate the whole population from this individual.")
        logger.info("Try to avoid duplicates.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual, iteration_callback)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode9.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_shuffle_mutation_operations()
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_fraction_iteration()

            self.population.es_sort_population()
            single_ind: ESIndividual = self.population.population[0]

            if single_ind.fitness <= self.population.target_fitness:
                self.population.es_early_exit(i)
                break

            self.population.population = [single_ind]
            current_size: int = 1

            new_ind: ESIndividual = single_ind.es_clone_internal()
            loop_counter: int = 0

            while current_size < self.population.population_size:
                for _ in range(self.population.num_of_mutations):
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
                    loop_counter = 0
                else:
                    # To prevent endless loops:
                    loop_counter += 1
                    if loop_counter >= 100:
                        self.population.population.append(new_ind.es_clone_internal())
                        current_size += 1
                        loop_counter = 0

        self.population.es_sort_population()
        self.population.es_after_iteration()
        self.population.es_calculate_fitness2()
        self.population.es_log_statistics()
        return self.population.es_get_best()

