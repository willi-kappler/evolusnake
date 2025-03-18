# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
from typing import override
import random as rnd
import unittest

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_population import ESPopulation
from evolusnake.es_individual import ESIndividual

from tests.common import TestIndividual


class TestPopulation(unittest.TestCase):
    def test_init(self):
        """
        Test normal initialisazion.
        """

        pass

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
        self.assertEqual(population1.reset_population, config1.reset_population)
        self.assertAlmostEqual(population1.target_fitness, config1.target_fitness)
        self.assertEqual(population1.best_index, 0)
        self.assertAlmostEqual(population1.best_fitness, 0.0)
        self.assertEqual(population1.worst_index, 0)
        self.assertAlmostEqual(population1.worst_fitness, 0.0)

        self.assertEqual(ind1.mutate_called, 0)
        self.assertEqual(ind1.calc_fitness_called, 0)
        self.assertEqual(ind1.clone_called, 10)
        self.assertEqual(ind1.randomize_called, 0)

        for ind in population1.population:
            self.assertEqual(ind.mutate_called, 40) # type: ignore
            self.assertEqual(ind.calc_fitness_called, 1) # type: ignore
            self.assertEqual(ind.clone_called, 0) # type: ignore
            self.assertEqual(ind.randomize_called, 1) # type: ignore

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
        config1.reset_population = True
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.reset_population, True)

        ind1.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        population1.es_reset_or_accept_best(ind1)
        population1.es_sort_population()

        self.assertGreater(population1.population[0].fitness, 0.0)

    def test_reset_or_accept_best2(self):
        """
        Test resetting the population or accepting the best solution.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.reset_population = False
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulation = ESPopulation(config1, ind1)
        self.assertEqual(population1.reset_population, False)

        ind1.data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ind1.es_calculate_fitness()

        population1.es_reset_or_accept_best(ind1)
        population1.es_sort_population()

        self.assertAlmostEqual(population1.population[0].fitness, 0.0)

if __name__ == "__main__":
    unittest.main()

