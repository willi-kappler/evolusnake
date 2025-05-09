# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 7.
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


class ESPopulationNode7(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 7")
        logger.info("Similar to population 1: clone and sort the whole population.")
        logger.info("The top worse half is overwritten, but mutate all individuals only once.")
        logger.info("Keep track of the best fitness and if it stays the same for too long, then")
        logger.info("randomize the whole population.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual, iteration_callback)
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode7.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_random_population()
        self.population.es_sort_population()
        self.population.es_set_num_iterations()

        offset: int = int(self.population.population_size / 2)
        previous_best_fitness: float = self.population.population[0].fitness
        previous_best_counter: int = 0
        best_fitness: float = previous_best_fitness
        iter_counter: int = 0

        while True:
            self.population.es_half_iteration()

            # Create a copy of each individual before mutating it:
            for j in range(offset):
                ind: ESIndividual = self.population.population[j]
                self.population.population[j + offset] = ind.es_clone_internal()

                ind.es_mutate_internal(self.population.es_get_mut_op())
                ind.es_calculate_fitness()

            self.population.es_sort_population()

            if self.population.population[0].fitness <= self.population.target_fitness:
                self.population.es_early_exit(iter_counter)
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

        self.population.es_after_iteration()
        self.population.es_log_statistics()
        logger.debug(f"{iter_counter=}")
        return self.population.es_get_best()

