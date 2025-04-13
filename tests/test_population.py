# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import unittest

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_population import ESPopulation
from evolusnake.es_individual import ESIndividual

from tests.common import TestIndividual


class TestPopulation(unittest.TestCase):
    def test_population_valid_config(self):
        """
        Test population init with a valid config.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)

        self.assertEqual(population1.population_size, config1.node_population_size)
        self.assertEqual(len(population1.population), config1.node_population_size)
        self.assertEqual(population1.num_of_iterations, config1.num_of_iterations)
        self.assertEqual(population1.num_of_mutations, config1.num_of_mutations)
        self.assertEqual(population1.accept_new_best, config1.accept_new_best)
        self.assertEqual(population1.randomize_population, config1.randomize_population)
        self.assertAlmostEqual(population1.target_fitness, config1.target_fitness)
        self.assertEqual(population1.best_index, 0)
        self.assertAlmostEqual(population1.best_fitness, 0.0)
        self.assertEqual(population1.worst_index, 0)
        self.assertAlmostEqual(population1.worst_fitness, 0.0)

        self.assertEqual(population1.max_iterations, config1.num_of_iterations)
        self.assertEqual(population1.half_iterations, int(population1.max_iterations / 2))
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.increase_iteration, config1.increase_iteration)
        self.assertEqual(population1.increase_mutation, config1.increase_mutation)

        self.assertEqual(population1.best_index, 0)
        self.assertAlmostEqual(population1.best_fitness, 0.0)
        self.assertEqual(population1.worst_index, 0)
        self.assertAlmostEqual(population1.worst_fitness, 0.0)

        self.assertEqual(population1.mutation_operations, config1.mutation_operations)
        self.assertEqual(population1.mutation_operations_len, len(population1.mutation_operations))
        self.assertFalse(population1.minimum_found)

        self.assertEqual(ind1.mutate_called, 0)
        self.assertEqual(ind1.calc_fitness_called, 0)
        self.assertEqual(ind1.clone_called, 10)
        self.assertEqual(ind1.randomize_called, 0)
        self.assertEqual(len(ind1.mut_op_counter), 0)

        for ind in population1.population:
            self.assertEqual(ind.mutate_called, 0)  # type: ignore
            self.assertEqual(ind.calc_fitness_called, 1)  # type: ignore
            self.assertEqual(ind.clone_called, 0)  # type: ignore
            self.assertEqual(ind.randomize_called, 1)  # type: ignore
            self.assertEqual(len(ind.mut_op_counter), 0)

    def test_population_invalid_config1(self):
        """
        Test population init with an invalid config: population size.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.node_population_size = 0
        ind1: TestIndividual = TestIndividual()

        with self.assertRaises(ValueError):
            population: ESPopulation = ESPopulation(config1, ind1)
            del population

    def test_population_invalid_config2(self):
        """
        Test population init with an invalid config: number of iteration.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.num_of_iterations = 0
        ind1: TestIndividual = TestIndividual()

        with self.assertRaises(ValueError):
            population: ESPopulation = ESPopulation(config1, ind1)
            del population

    def test_population_invalid_config3(self):
        """
        Test population init with an invalid config: number of mutations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.num_of_mutations = 0
        ind1: TestIndividual = TestIndividual()

        with self.assertRaises(ValueError):
            population: ESPopulation = ESPopulation(config1, ind1)
            del population

    def test_population_find_worst_individual(self):
        """
        Test finding the worst individual in a population.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_worst_individual()

        for i in range(population1.population_size):
            self.assertGreaterEqual(population1.worst_fitness, population1.population[i].fitness)

        self.assertAlmostEqual(population1.worst_fitness, population1.population[population1.worst_index].fitness)

    def test_population_find_best_and_worst_individual(self):
        """
        Test finding the best and worst individual in a population.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_best_and_worst_individual()

        for i in range(population1.population_size):
            self.assertGreaterEqual(population1.worst_fitness, population1.population[i].fitness)

        for i in range(population1.population_size):
            self.assertLessEqual(population1.best_fitness, population1.population[i].fitness)

        self.assertAlmostEqual(population1.best_fitness, population1.population[population1.best_index].fitness)
        self.assertAlmostEqual(population1.worst_fitness, population1.population[population1.worst_index].fitness)

    def test_sort_population(self):
        """
        Test sorting the population.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_sort_population()

        for i in range(population1.population_size - 1):
            self.assertLessEqual(population1.population[i].fitness, population1.population[i + 1].fitness)

    def test_reset_or_accept_best1(self):
        """
        Test resetting the population or accepting the best solution.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.randomize_population = True
        config1.accept_new_best = True
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.randomize_population, True)
        self.assertEqual(population1.accept_new_best, True)

        ind1.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        best_counter: int = 0
        for _ in range(10):
            population1.es_randomize_or_accept_best(ind1)
            population1.es_sort_population()
            if population1.population[0].fitness == 0.0:
                best_counter += 1

        self.assertLess(best_counter, 9)

    def test_reset_or_accept_best2(self):
        """
        Test resetting the population or accepting the best solution.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.randomize_population = False
        config1.accept_new_best = True
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.randomize_population, False)
        self.assertEqual(population1.accept_new_best, True)

        ind1.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ind1.es_calculate_fitness()

        best_counter: int = 0
        for _ in range(10):
            population1.es_randomize_or_accept_best(ind1)
            population1.es_sort_population()
            if population1.population[0].fitness == 0.0:
                best_counter += 1

        self.assertEqual(best_counter, 10)

    def test_increate_iteration_mutation1(self):
        """
        Test increasing the number of mutations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.increase_mutation = 1
        config1.increase_iteration = 0
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.increase_mutation, 1)
        self.assertEqual(population1.increase_iteration, 0)
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations + 1)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations + 2)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations)

    def test_increate_iteration_mutation2(self):
        """
        Test increasing the number of iterations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.increase_mutation = 0
        config1.increase_iteration = 1
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.increase_mutation, 0)
        self.assertEqual(population1.increase_iteration, 1)
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations + 1)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations + 2)

    def test_increate_iteration_mutation3(self):
        """
        Test increasing the number of mutations and iterations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.increase_mutation = 2
        config1.increase_iteration = 2
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.increase_mutation, 2)
        self.assertEqual(population1.increase_iteration, 2)
        self.assertEqual(population1.max_mutations, config1.num_of_mutations)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations + 2)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations + 2)

        population1.es_increase_iteration_mutation()
        self.assertEqual(population1.max_mutations, config1.num_of_mutations + 4)
        self.assertEqual(population1.max_iterations, config1.num_of_iterations + 4)

    def test_set_num_mutations1(self):
        """
        Test randomly changing the number of mutations.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        self.assertEqual(population1.num_of_mutations, config1.num_of_mutations)

        for _ in range(10):
            population1.es_set_num_mutations()
            self.assertGreaterEqual(population1.num_of_mutations, 1)
            self.assertLessEqual(population1.num_of_iterations, config1.num_of_iterations)

    def test_set_num_mutations2(self):
        """
        Test randomly changing the number of mutations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.num_of_mutations = 1
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.num_of_mutations, 1)

        for _ in range(10):
            population1.es_set_num_mutations()
            self.assertEqual(population1.num_of_mutations, 1)

    def test_set_num_iterations(self):
        """
        Test randomly changing the number of iterations.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        self.assertEqual(population1.num_of_iterations, config1.num_of_iterations)

        for _ in range(10):
            population1.es_set_num_iterations()
            self.assertGreaterEqual(population1.num_of_iterations, population1.half_iterations)
            self.assertLessEqual(population1.num_of_iterations, config1.num_of_iterations)

    def test_randomize_worst(self):
        """
        Test randomizing the worst individual.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_worst_individual()
        current_worst_fitness: float = population1.worst_fitness
        current_worst_index: int = population1.worst_index

        equal_counter: int = 0

        for _ in range(10):
            population1.es_randomize_worst()
            if current_worst_fitness == population1.population[population1.worst_index].fitness:
                equal_counter += 1
            self.assertEqual(current_worst_index, population1.worst_index)

        self.assertLess(equal_counter, 9)

    def test_replace_best(self):
        """
        Test replacing the best individual with a better one.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        ind2: TestIndividual = TestIndividual()
        ind2.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ind2.fitness = 0.0
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_best_and_worst_individual()
        current_best_index: int = population1.best_index
        population1.population[current_best_index].fitness = 1.0
        current_best_fitness: float = population1.best_fitness
        population1.es_replace_best(ind2)
        self.assertGreaterEqual(current_best_fitness, population1.best_fitness)
        self.assertEqual(current_best_index, population1.best_index)

    def test_replace_worst(self):
        """
        Test replacing the worst individual with a better one.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        ind2: TestIndividual = TestIndividual()
        ind2.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ind2.fitness = 0.0
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_best_and_worst_individual()
        current_worst_fitness: float = population1.worst_fitness
        current_worst_index: int = population1.worst_index
        population1.es_replace_worst(ind2)
        self.assertGreater(current_worst_fitness, population1.population[population1.worst_index].fitness)
        self.assertEqual(current_worst_index, population1.worst_index)

    def test_clone_best_to_worst(self):
        """
        Test replacing the worst individual with the best one.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_best_and_worst_individual()
        population1.es_clone_best_to_worst()
        self.assertEqual(population1.best_fitness, population1.population[population1.worst_index].fitness)

    def test_get_best(self):
        """
        Test getting the best individual.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_find_best_and_worst_individual()
        ind2: ESIndividual = population1.es_get_best()
        self.assertEqual(population1.best_fitness, ind2.fitness)

    def test_get_mut_op1(self):
        """
        Test getting a random mutation operation.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.mutation_operations = []
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        for _ in range(10):
            self.assertEqual(population1.es_get_mut_op(), 0)

    def test_get_mut_op2(self):
        """
        Test getting a random mutation operation.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.mutation_operations = [3]
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        for _ in range(10):
            self.assertEqual(population1.es_get_mut_op(), 3)

    def test_get_mut_op3(self):
        """
        Test getting a random mutation operation.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.mutation_operations = [3, 8, 17, 25]
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        for _ in range(10):
            self.assertIn(population1.es_get_mut_op(), config1.mutation_operations)

    def test_early_exit(self):
        """
        Test early exit.
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        self.assertFalse(population1.minimum_found)
        population1.es_early_exit(1)
        self.assertTrue(population1.minimum_found)

    def test_log_statistics(self):
        """
        Test logging of statistics
        """

        config1: ESConfiguration = ESConfiguration()
        ind1: TestIndividual = TestIndividual()
        population1: ESPopulation = ESPopulation(config1, ind1)

        population1.es_log_statistics()

        # TODO: Find a way to check logs via assert.


if __name__ == "__main__":
    unittest.main()

