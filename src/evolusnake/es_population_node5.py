# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 5.
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


class ESPopulationNode5(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 5")
        logger.debug("Calculate the average fitness. If after mutation the individual is")
        logger.debug("better than the average keep it. Replace the worst with the second worst.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual, iteration_callback)
        self.population.es_sort_population()
        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1
        self.es_calc_average_fitness()

    def es_calc_average_fitness(self):
        best: float = self.population.population[0].fitness
        worst: float = self.population.population[-1].fitness
        self.average_fitness = (best + worst) / 2.0

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode5.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()
        self.population.es_set_num_mutations()
        self.population.minimum_found = False
        second_worst: ESIndividual = self.population.population[-2].es_clone_internal()

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_half_iteration()

            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate_internal(self.population.es_get_mut_op())
                tmp_ind.es_calculate_fitness()

                self.population.es_check_limit(tmp_ind, self.average_fitness, j)

                if tmp_ind.fitness <= self.population.target_fitness:
                    self.population.es_early_exit(i)
                    break

            if self.population.minimum_found:
                break

            self.population.es_sort_population()
            second_worst = self.population.population[-2].es_clone_internal()
            self.population.population[-1] = second_worst
            self.es_calc_average_fitness()

        self.population.es_after_iteration()
        self.population.es_log_statistics()
        return self.population.es_get_best()

