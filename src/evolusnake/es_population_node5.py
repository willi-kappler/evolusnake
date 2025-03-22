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
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)

class ESPopulationNode5(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 5")
        logger.debug("TODO...")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual)
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
        minimum_found: bool = False

        for i in range(self.population.num_of_iterations):
            for j in range(self.population.population_size):
                tmp_ind: ESIndividual = self.population.population[j].es_clone()

                for _ in range(self.population.num_of_mutations):
                    tmp_ind.es_mutate()
                tmp_ind.es_calculate_fitness()

                if tmp_ind.fitness < self.average_fitness:
                    self.population.population[j] = tmp_ind

                    if tmp_ind.fitness <= self.population.target_fitness:
                        logger.info(f"Early exit at iteration {i}")
                        minimum_found = True
                        break

            if minimum_found:
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

            self.population.es_sort_population()
            second_worst = self.population.population[-2].es_clone()
            self.population.population[-1] = second_worst
            self.es_calc_average_fitness()

        best_fitness: float = self.population.population[0].fitness
        worst_fitness: float = self.population.population[-1].fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")
        logger.debug(f"{self.average_fitness=}")

        return self.population.es_get_best()

