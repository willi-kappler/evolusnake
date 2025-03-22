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
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)

class ESPopulationNode3(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 3")
        logger.debug("TODO...")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population: ESPopulation = ESPopulation(config, individual)

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode3.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_find_best_and_worst_individual()

        max_iter = self.population.num_of_iterations * self.population.population_size

        for i in range(max_iter):
            j = rnd.randrange(self.population.population_size)
            tmp_ind: ESIndividual = self.population.population[j].es_clone()

            for _ in range(self.population.num_of_mutations):
                tmp_ind.es_mutate()
            tmp_ind.es_calculate_fitness()

            if tmp_ind.fitness < self.population.best_fitness:
                self.population.es_replace_best(tmp_ind)
                if tmp_ind.fitness <= self.population.target_fitness:
                    logger.info(f"Early exit at iteration {i}")
                    break
            elif tmp_ind.fitness < self.population.worst_fitness:
                self.population.es_replace_worst(tmp_ind)
                self.population.es_find_worst_individual()

            # Change mutation rate:
            self.population.es_set_num_mutations()

        best_fitness: float = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")

        return self.population.es_get_best()

