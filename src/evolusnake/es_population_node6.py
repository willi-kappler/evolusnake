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
from evolusnake.es_population import ESPopulation, ESIterationCallBack

logger = logging.getLogger(__name__)


class ESPopulationNode6(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 6")
        logger.info("Clone two individuals for each in the population.")
        logger.info("After each mutation, calculate fitness and keep if better.")
        logger.info("The first clone keeps mutating, the second clone is reset to the initial individual.")
        logger.info("The best of all the mutations is kept and the next individual is mutated.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual, iteration_callback)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode6.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_shuffle_mutation_operations()
        self.population.minimum_found = False

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_fraction_iteration()

            for j in range(self.population.population_size):
                tmp_ind1: ESIndividual = self.population.population[j].es_clone_internal()
                initial_ind: ESIndividual = tmp_ind1.es_clone_internal()
                best_ind: ESIndividual = tmp_ind1.es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind1.es_mutate_internal(self.population.es_get_mut_op())
                    tmp_ind1.es_calculate_fitness()
                    if tmp_ind1.fitness < best_ind.fitness:
                        best_ind = tmp_ind1.es_clone_internal()

                    tmp_ind2: ESIndividual = initial_ind.es_clone_internal()
                    tmp_ind2.es_mutate_internal(self.population.es_get_mut_op())
                    tmp_ind2.es_calculate_fitness()
                    if tmp_ind2.fitness < best_ind.fitness:
                        best_ind = tmp_ind2.es_clone_internal()

                if best_ind.fitness < self.population.population[j].fitness:
                    self.population.population[j] = best_ind

                    if best_ind.fitness <= self.population.target_fitness:
                        self.population.es_early_exit(i)
                        break

            if self.population.minimum_found:
                break

        self.population.es_find_best_and_worst_individual()
        self.population.es_after_iteration()
        self.population.es_calculate_fitness2()
        self.population.es_log_statistics()
        return self.population.es_get_best()

