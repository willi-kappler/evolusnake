# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the base data class for a population.
"""

# Python std lib:
import logging
import random as rnd

# Local imports
from evolusnake.es_individual import ESIndividual
from evolusnake.es_config import ESConfiguration

logger = logging.getLogger(__name__)

class ESPopulation:
    def __init__(self, config: ESConfiguration, individual: ESIndividual):
        if config.node_population_size < 2:
            raise ValueError(f"Node population must be at least 2, {config.node_population_size}")

        if config.num_of_iterations < 1:
            raise ValueError(f"Number of iterations must be at least 1, {config.num_of_iterations}")

        if config.num_of_mutations < 1:
            raise ValueError(f"Number of mutations must be at least 1, {config.num_of_mutations}")

        self.population_size: int = config.node_population_size
        self.population: list[ESIndividual] = []

        for _ in range(self.population_size):
            ind: ESIndividual = individual.es_clone()
            ind.es_randomize()
            ind.es_calculate_fitness()
            self.population.append(ind)

        self.num_of_iterations: int = config.num_of_iterations
        self.num_of_mutations: int = config.num_of_mutations
        self.max_mutations: int = config.num_of_mutations
        self.accept_new_best: bool = config.accept_new_best
        self.randomize_population: bool = config.randomize_population
        self.target_fitness: float = config.target_fitness
        self.increase_iteration: int = config.increase_iteration
        self.increase_mutation: int = config.increase_mutation

        self.best_index: int = 0
        self.best_fitness: float = 0.0
        self.worst_index: int = 0
        self.worst_fitness: float = 0.0

        # Init random number generator:
        rnd.seed()

        logger.debug(f"{self.population_size=}, {self.target_fitness=}")
        logger.debug(f"{self.num_of_iterations=}, {self.num_of_mutations=}")
        logger.debug(f"{self.accept_new_best=}, {self.randomize_population=}")
        logger.debug(f"{self.increase_iteration=}, {self.increase_mutation=}")

    def es_find_worst_individual(self):
        self.worst_index = 0
        self.worst_fitness = self.population[0].fitness

        for i in range(1, self.population_size):
            fitness: float = self.population[i].fitness
            if fitness > self.worst_fitness:
                self.worst_fitness = fitness
                self.worst_index = i

    def es_find_best_and_worst_individual(self):
        self.worst_index = 0
        self.worst_fitness = self.population[0].fitness
        self.best_index = 0
        self.best_fitness = self.population[0].fitness

        for i in range(1, self.population_size):
            fitness: float = self.population[i].fitness
            if fitness > self.worst_fitness:
                self.worst_fitness = fitness
                self.worst_index = i
            elif fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_index = i

    def es_sort_population(self):
        self.population.sort(key=lambda ind: ind.fitness)

    def es_random_population(self):
        for ind in self.population:
            ind.es_randomize()
            ind.es_calculate_fitness()

    def es_randomize_or_accept_best(self, best: ESIndividual):
        if self.randomize_population:
            self.es_random_population()
        elif self.accept_new_best:
                self.population[0] = best

    def es_increase_iteration_mutation(self):
        if self.increase_iteration > 0:
            self.num_of_iterations += self.increase_iteration
            logger.debug(f"{self.num_of_iterations=}")

        if self.increase_mutation > 0:
            self.max_mutations += self.increase_mutation
            logger.debug(f"{self.max_mutations=}")

    def es_set_num_mutations(self):
        self.num_of_mutations -= 1
        if self.num_of_mutations <= 0:
            self.num_of_mutations = self.max_mutations

    def es_randomize_worst(self):
        worst = self.population[self.worst_index]
        worst.es_randomize()
        worst.es_calculate_fitness()

    def es_replace_best(self, individual: ESIndividual):
        self.population[self.best_index] = individual
        self.best_fitness = individual.fitness

