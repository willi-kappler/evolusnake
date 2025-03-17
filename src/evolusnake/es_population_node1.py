# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the base data class for an individual.
"""

# Python std lib:
import logging
from typing import Any,  override

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

        self.population = ESPopulation(config, individual)

        # Create working list with additional individuals:
        for _ in range(self.population.population_size):
            self.population.population.append(individual)

    @override
    def ps_process_data(self, data: Any) -> Any:
        logger.debug(f"ESPopulationNode1.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_reset_or_accept_best(data)
        offset = self.population.population_size
        max_mutation: int = self.population.num_of_mutations

        for i in range(self.population.num_of_iterations):
            # Create a copy of each individual before mutating it:
            for j in range(offset):
                ind: ESIndividual = self.population.population[j]
                self.population.population[j + offset] = ind.es_clone()

                # Now mutate the original individual:
                for _ in range(max_mutation):
                    ind.es_mutate()
                ind.es_calculate_fitness()

            self.population.es_sort_population()

            if self.population.population[0].fitness <= self.population.target_fitness:
                logger.info(f"Early exit at iteration {i}")
                break

            # Change mutation rate:
            max_mutation -= 1
            if max_mutation <= 0:
                max_mutation = self.population.num_of_mutations

        best_fitness: float = self.population.population[0].fitness
        worst_fitness: float = self.population.population[offset - 1].fitness
        logger.debug(f"{best_fitness=}, {worst_fitness=}")

        return self.population.population[0]

