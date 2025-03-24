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

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)

class ESPopulationNode4(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 4")
        logger.debug("Use a global fitness that is the same for all individuals.")
        logger.debug("Mutate an individual and if it's better than the global fitness keep it.")
        logger.debug("Reduce global fitness each iteration.")
        logger.debug("If no individual is better, increase the global fitness a bit.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual)
        self.population.es_find_worst_individual()

        self.global_fitness = self.population.worst_fitness

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode4.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        minimum_found: bool = False
        ind_below_global: int = 0
        all_above_global: int = 0

        self.population.es_set_num_iterations()
        logger.debug(f"Iterations: {self.population.num_of_iterations}")

        for i in range(self.population.num_of_iterations):
            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate()
                tmp_ind.es_calculate_fitness()

                if tmp_ind.fitness < self.global_fitness:
                    self.population.population[j] = tmp_ind

                    if tmp_ind.fitness <= self.population.target_fitness:
                        logger.info(f"Early exit at iteration {i}")
                        minimum_found = True
                        break

            if minimum_found:
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

            ind_below_global = 0
            for ind in self.population.population:
                if ind.fitness < self.global_fitness:
                    ind_below_global += 1

            if ind_below_global >= 2:
                self.global_fitness = self.global_fitness * 0.9
            else:
                self.global_fitness = self.global_fitness * 1.01
                all_above_global += 1

        self.population.es_find_best_and_worst_individual()

        best_fitness: float = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")
        logger.debug(f"{self.global_fitness=}, {all_above_global=}")

        self.population.es_clone_best_to_worst()

        return self.population.es_get_best()

