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
import time

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
        self.max_iterations: int = config.num_of_iterations
        self.half_iterations: int = int(self.max_iterations / 2)
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

        self.mutation_operations: list = config.mutation_operations
        self.mutation_operations_len: int = len(config.mutation_operations)
        self.minimum_found: bool = False

        # Init random number generator:
        rnd.seed()

        logger.debug(f"{self.population_size=}, {self.target_fitness=}")
        logger.debug(f"{self.num_of_iterations=}, {self.num_of_mutations=}")
        logger.debug(f"{self.accept_new_best=}, {self.randomize_population=}")
        logger.debug(f"{self.increase_iteration=}, {self.increase_mutation=}")
        logger.debug(f"{self.mutation_operations=}")

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
            ind.es_reset_counter()
            ind.es_randomize()
            ind.es_calculate_fitness()

    def es_randomize_or_accept_best(self, best: ESIndividual):
        if self.randomize_population:
            self.es_random_population()
        elif self.accept_new_best:
            i: int = rnd.randrange(self.population_size)
            self.population[i] = best

    def es_increase_iteration_mutation(self):
        if self.increase_iteration > 0:
            self.max_iterations += self.increase_iteration
            logger.debug(f"{self.max_iterations=}")

        if self.increase_mutation > 0:
            self.max_mutations += self.increase_mutation
            logger.debug(f"{self.max_mutations=}")

    def es_set_num_mutations(self):
        if self.max_mutations > 1:
            self.num_of_mutations = rnd.randrange(self.max_mutations) + 1
        else:
            self.num_of_mutations = 1

    def es_set_num_iterations(self):
        self.num_of_iterations = rnd.randrange(self.half_iterations, self.max_iterations + 1)
        logger.debug(f"Iterations: {self.num_of_iterations}")

    def es_randomize_worst(self):
        worst = self.population[self.worst_index]
        worst.es_randomize()
        worst.es_calculate_fitness()
        # Now maybe no longer the worst!

    def es_replace_best(self, individual: ESIndividual):
        if individual.fitness < self.best_fitness:
            self.population[self.best_index] = individual
            self.best_fitness = individual.fitness
        # Still the best at index: self.best_index!

    def es_replace_worst(self, individual: ESIndividual):
        self.population[self.worst_index] = individual
        # Now no longer the worst!

    def es_clone_best_to_worst(self):
        ind = self.population[self.best_index].es_clone()
        self.population[self.worst_index] = ind
        # Now no longer the worst!

    def es_get_best(self) -> ESIndividual:
        return self.population[self.best_index]

    def es_get_mut_op(self) -> int:
        match self.mutation_operations_len:
            case 0:
                return 0
            case 1:
                return self.mutation_operations[0]
            case _:
                i = rnd.randrange(self.mutation_operations_len)
                return self.mutation_operations[i]

    def es_check_limit(self, ind: ESIndividual, limit: float, i: int):
        if (ind.fitness < limit) or (ind.fitness < self.population[i].fitness):
            self.population[i] = ind

    def es_early_exit(self, iteration: int):
        logger.info(f"Early exit at iteration {iteration}")
        self.minimum_found = True

        if iteration == 0:
            # Wait some seconds to avoid spamming the server.
            time.sleep(5)

    def es_log_statistics(self):
        best_individual: ESIndividual = self.population[self.best_index]
        best_fitness: float = best_individual.fitness
        worst_individual: ESIndividual = self.population[self.worst_index]
        worst_fitness: float = worst_individual.fitness
        actual_best: float = best_individual.es_actual_fitness()
        actual_worst: float = worst_individual.es_actual_fitness()
        logger.debug(f"{best_fitness=}, {worst_fitness=}")
        logger.debug(f"{actual_best=}, {actual_worst=}")
        logger.debug(f"Best individual mutations: {best_individual.mut_op_counter}")
        logger.debug(f"Worst individual mutations: {worst_individual.mut_op_counter}")


