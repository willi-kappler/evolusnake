# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 4.
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
from evolusnake.es_population import ESIterationCallBack, ESPopulation

logger = logging.getLogger(__name__)


class ESPopulationNode4(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack = ESIterationCallBack()):
        logger.info("Init population node type 4")
        logger.debug("Use a global fitness that is the same for all individuals.")
        logger.debug("Mutate an individual and if it's better than the global fitness keep it.")
        logger.debug("Reduce global fitness each iteration.")
        logger.debug("If no individual is better, increase the global fitness a bit.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual, iteration_callback)
        self.population.es_find_worst_individual()

        self.global_fitness = self.population.es_get_worst_fitness()

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode4.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()
        self.population.es_set_num_mutations()
        self.population.minimum_found = False
        ind_below_global: int = 0
        all_above_global: int = 0
        min_num_ind: int = rnd.randrange(1, int(self.population.population_size / 2))

        logger.debug(f"{min_num_ind=}")

        self.population.es_before_iteration()

        for i in range(self.population.num_of_iterations):
            self.population.es_fraction_iteration()

            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate_internal(self.population.es_get_mut_op())
                tmp_ind.es_calculate_fitness()

                self.population.es_check_limit(tmp_ind, self.global_fitness, j)

                if tmp_ind.fitness <= self.population.target_fitness:
                    self.population.es_early_exit(i)
                    break

            if self.population.minimum_found:
                break

            ind_below_global = 0
            for ind in self.population.population:
                if ind.fitness < self.global_fitness:
                    ind_below_global += 1

            if ind_below_global >= min_num_ind:
                self.global_fitness = self.global_fitness * 0.9
            else:
                self.global_fitness = self.global_fitness * 1.01
                all_above_global += 1

        logger.debug(f"{all_above_global=}")

        self.population.es_find_best_and_worst_individual()
        self.population.es_after_iteration()
        self.population.es_calculate_fitness2()
        self.population.es_log_statistics()
        self.population.es_clone_best_to_worst()
        return self.population.es_get_best()

