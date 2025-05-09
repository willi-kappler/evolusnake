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
from evolusnake.es_population import ESPopulation, ESIterationCallBack

logger = logging.getLogger(__name__)


class ESPopulationNode2(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 2")
        logger.info("Mutate a clone and if it's better than the previous version keep it.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual, iteration_callback)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode2.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()
        self.population.es_set_num_mutations()
        self.population.minimum_found = False

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_half_iteration()

            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate_internal(self.population.es_get_mut_op())
                tmp_ind.es_calculate_fitness()

                if tmp_ind.fitness < self.population.population[j].fitness:
                    self.population.population[j] = tmp_ind

                    if tmp_ind.fitness <= self.population.target_fitness:
                        self.population.es_early_exit(i)
                        break

            if self.population.minimum_found:
                break

        self.population.es_find_best_and_worst_individual()
        self.population.es_after_iteration()
        self.population.es_log_statistics()
        return self.population.es_get_best()

