# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

from typing import override
import random as rnd
import unittest

from evolusnake.es_config import ESConfiguration
from evolusnake.es_population import ESPopulation
from evolusnake.es_individual import ESIndividual


class TestIndividual(ESIndividual):
    def __init__(self):
        self.mutate_called: int = 0
        self.calc_fitness_called: int = 0
        self.clone_called: int = 0
        self.randomize_called: int = 0

        self.data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.data_size = 10

    @override
    def es_mutate(self):
        pos = rnd.randrange(self.data_size)
        self.data[pos] = 1 - self.data[pos]
        self.mutate_called += 1

    @override
    def es_randomize(self):
        for _ in range(self.data_size):
            self.es_mutate()

        self.randomize_called += 1

    @override
    def es_calculate_fitness(self):
        self.fitness = float(sum(self.data))
        self.calc_fitness_called += 1

    @override
    def es_clone(self) -> "TestIndividual":
        new: TestIndividual = TestIndividual()
        new.data = self.data[:]
        new.data_size = self.data_size

        self.clone_called += 1

        return new


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

if __name__ == "__main__":
    unittest.main()

