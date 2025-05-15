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


class ESIterationCallBack:
    def __init__(self):
        pass

    def es_get_iteration_factor(self) -> int:
        # This method sets the fraction at which es_fraction_iteration()
        # will be called. A return value of 2 means at 50%
        # A value of 3 means 33.33% and 66.66% and so on.
        return 1

    def es_before_iteration(self, population: "ESPopulation"):
        # This method is called just before the iteration starts.
        pass

    def es_fraction_iteration(self, population: "ESPopulation"):
        # This method is called when half of the iteration counter is reached.
        pass

    def es_after_of_iteration(self, population: "ESPopulation"):
        # This method is called after the end of the whole iteration.
        pass


class ESPopulation:
    def __init__(self, config: ESConfiguration, individual: ESIndividual,
            iteration_callback: ESIterationCallBack):
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
        self.target_fitness2: float = config.target_fitness2
        self.increase_iteration: int = config.increase_iteration
        self.increase_mutation: int = config.increase_mutation

        self.best_index: int = 0
        self.worst_index: int = 0

        self.mutation_operations: list = config.mutation_operations
        self.mutation_operations_len: int = len(config.mutation_operations)
        self.minimum_found: bool = False

        self.iteration_callback = iteration_callback
        self.iteration_counter: int = 0
        self.fraction_value: int = self.iteration_callback.es_get_iteration_factor()
        self.fraction_iterations: int = int(self.num_of_iterations / self.fraction_value)

        # Init random number generator:
        rnd.seed()

        logger.debug(f"{self.population_size=}, {self.target_fitness=}, {self.target_fitness2=}")
        logger.debug(f"{self.num_of_iterations=}, {self.num_of_mutations=}")
        logger.debug(f"{self.accept_new_best=}, {self.randomize_population=}")
        logger.debug(f"{self.increase_iteration=}, {self.increase_mutation=}")
        logger.debug(f"{self.mutation_operations=}")

    def es_find_worst_individual(self):
        self.worst_index = 0
        worst_fitness: float = self.population[0].fitness

        for i in range(1, self.population_size):
            fitness: float = self.population[i].fitness
            if fitness > worst_fitness:
                worst_fitness = fitness
                self.worst_index = i

    def es_find_best_and_worst_individual(self):
        self.worst_index = 0
        worst_fitness: float = self.population[0].fitness
        self.best_index = 0
        best_fitness: float = self.population[0].fitness

        for i in range(1, self.population_size):
            fitness: float = self.population[i].fitness
            if fitness > worst_fitness:
                worst_fitness = fitness
                self.worst_index = i
            elif fitness < best_fitness:
                best_fitness = fitness
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
            self.population[i].es_from_server(best)

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
        self.iteration_counter = 0
        self.fraction_iterations = int(self.num_of_iterations / self.fraction_value)
        logger.debug(f"Iterations: {self.num_of_iterations}")

    def es_randomize_worst(self):
        worst = self.population[self.worst_index]
        worst.es_randomize()
        worst.es_calculate_fitness()
        # Now maybe no longer the worst!

    def es_replace_best(self, individual: ESIndividual):
        if individual.fitness < self.population[self.best_index].fitness:
            self.population[self.best_index] = individual
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

    def es_get_best_fitness(self) -> float:
        return self.population[self.best_index].fitness

    def es_get_worst_fitness(self) -> float:
        return self.population[self.worst_index].fitness

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

    def es_calculate_fitness2_2(self):
        fitness2_list: list = []

        for i in range(self.population_size):
            ind: ESIndividual = self.population[i]
            if ind.fitness < 0.01:
                ind.es_calculate_fitness2()
                fitness2_list.append((ind.fitness2, i))

        if fitness2_list:
            fitness2_list.sort(key=lambda x: x[0])
            index: int = fitness2_list[0][1]

            self.population[self.best_index] = self.population[index]

    def es_calculate_fitness2(self):
        ind = self.population[self.best_index]

        if ind.fitness < 0.01:
            ind.es_calculate_fitness2()

    def es_log_statistics(self):
        best_individual: ESIndividual = self.population[self.best_index]
        best_fitness: float = best_individual.fitness
        best_fitness2: float = best_individual.fitness2
        worst_individual: ESIndividual = self.population[self.worst_index]
        worst_fitness: float = worst_individual.fitness
        worst_fitness2: float = worst_individual.fitness2
        actual_best: float = best_individual.es_actual_fitness()
        actual_worst: float = worst_individual.es_actual_fitness()
        logger.debug(f"{best_fitness=}, {best_fitness2=}")
        logger.debug(f"{worst_fitness=}, {worst_fitness2=}")
        logger.debug(f"{actual_best=}, {actual_worst=}")
        logger.debug(f"Best individual mutations: {best_individual.mut_op_counter}")
        logger.debug(f"Worst individual mutations: {worst_individual.mut_op_counter}")

        best_individual.es_new_best_individual()

    def es_before_iteration(self):
        self.iteration_callback.es_before_iteration(self)

    def es_fraction_iteration(self):
        self.iteration_counter += 1
        if self.iteration_counter > self.fraction_iterations:
            self.iteration_counter = 0
            self.iteration_callback.es_fraction_iteration(self)

    def es_after_iteration(self):
        self.iteration_callback.es_after_of_iteration(self)


