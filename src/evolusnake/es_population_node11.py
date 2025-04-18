# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the class for population type 9.
"""

# Python std lib:
import logging
from typing import override
import math
import random as rnd

# External imports:
from parasnake.ps_node import PSNode

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESPopulation

logger = logging.getLogger(__name__)


class ESPopulationNode11(PSNode):
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        logger.info("Init population node type 11")
        logger.info("Use a sine wave for the fitness limit.")

        super().__init__(config.parasnake_config)
        logger.debug(f"Node ID: {self.node_id}")

        self.population = ESPopulation(config, individual)
        self.sine_base: float = 0.0
        self.sine_amplitude: float = 0.0

    @override
    def ps_process_data(self, data: ESIndividual) -> ESIndividual:
        logger.debug("ESPopulationNode11.ps_process_data()")
        logger.debug(f"Individual from server: {data.fitness}")

        self.population.es_randomize_or_accept_best(data)
        self.population.es_increase_iteration_mutation()
        self.population.es_set_num_iterations()
        self.population.minimum_found = False

        sine_freq: float = math.tau / float(self.population.num_of_iterations)
        logger.debug(f"{sine_freq=}")
        current_limit: float = 0.0

        for i in range(self.population.num_of_iterations):
            current_limit = self.sine_base + (self.sine_amplitude * math.sin(sine_freq * i))

            for j in range(self.population.population_size):
                ind: ESIndividual = self.population.population[j].es_clone_internal()

                for _ in range(self.population.num_of_mutations):
                    ind.es_mutate_internal(self.population.es_get_mut_op())
                ind.es_calculate_fitness()

                self.population.es_check_limit(ind, current_limit, j)

                # if ind.fitness < current_limit:
                #     self.population.population[j] = ind
                # elif ind.fitness < self.population.population[j].fitness:
                #     self.population.population[j] = ind

                if ind.fitness < self.population.target_fitness:
                    self.population.es_early_exit(i)
                    break

            if self.population.minimum_found:
                break

            # Change mutation rate:
            self.population.es_set_num_mutations()

        self.population.es_find_best_and_worst_individual()

        mid_fitness: float = (self.population.best_fitness + self.population.worst_fitness) / 2.0
        self.sine_base = self.population.best_fitness
        self.sine_amplitude = mid_fitness - self.population.best_fitness
        logger.debug(f"{mid_fitness=}, {self.sine_base=}, {self.sine_amplitude=}")

        self.population.es_log_statistics()
        return self.population.es_get_best()

