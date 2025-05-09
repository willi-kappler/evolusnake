# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 3.
"""

# Python std lib:
import logging
import random as rnd
from typing import override

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation, ESIterationCallBack

logger = logging.getLogger(__name__)


class ESPopulationNode3(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 3")
        logger.debug("Randomly pick an individual and mutate it.")
        logger.debug("If it's better than the best replace it.")
        logger.debug("Else if it's better than the worst replace it.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual, iteration_callback)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode3.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_find_best_and_worst_individual()
        self.population.es_set_num_iterations()
        self.population.es_set_num_mutations()

        max_iter = self.population.num_of_iterations * self.population.population_size

        self.population.es_before_iteration()

        for i in range(max_iter):
            self.population.es_fraction_iteration()

            j = rnd.randrange(self.population.population_size)
            tmp_ind: ESIndividual = self.population.population[j].es_clone_internal()

            for _ in range(self.population.num_of_mutations):
                tmp_ind.es_mutate_internal(self.population.es_get_mut_op())
            tmp_ind.es_calculate_fitness()

            if tmp_ind.fitness < self.population.best_fitness:
                self.population.es_replace_best(tmp_ind)
                if tmp_ind.fitness <= self.population.target_fitness:
                    self.population.es_early_exit(i)
                    break
            elif tmp_ind.fitness < self.population.worst_fitness:
                self.population.es_replace_worst(tmp_ind)
                self.population.es_find_worst_individual()

        self.population.es_after_iteration()
        self.population.es_log_statistics()
        return self.population.es_get_best()

