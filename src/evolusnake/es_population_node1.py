# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 1.
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

class ESPopulationNode1(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 1")
        logger.info("Clone population and mutate individuals in place. Then sort population by fitness.")
        logger.info("The worst individuals are overwritten.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)

        # Create working list with additional individuals:
        for _ in range(self.population.population_size):
            self.population.population.append(individual)

        self.population.best_index = 0
        self.population.worst_index = self.population.population_size - 1

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode1.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        offset = self.population.population_size

        for i in range(self.population.num_of_iterations):
            # Create a copy of each individual before mutating it:
            for j in range(offset):
                ind: ESIndividual = self.population.population[j]
                self.population.population[j + offset] = ind.es_clone()

                # Now mutate the original individual:
                for _ in range(self.population.num_of_mutations):
                    ind.es_mutate()
                ind.es_calculate_fitness()

            self.population.es_sort_population()

            if self.population.population[0].fitness <= self.population.target_fitness:
                logger.info(f"Early exit at iteration {i}")
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

        self.population.best_fitness = self.population.population[0].fitness
        self.population.worst_fitness = self.population.population[offset - 1].fitness

        best_fitness: float = self.population.best_fitness
        worst_fitness: float = self.population.worst_fitness

        logger.debug(f"{best_fitness=}, {worst_fitness=}")

        return self.population.es_get_best()

