# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import unittest

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_population_node6 import ESPopulationNode6
from evolusnake.es_individual import ESIndividual

from tests.common import TestIndividual

# External imports:
from parasnake.ps_config import PSConfiguration


class TestPopulation(unittest.TestCase):
    def test_population_process_data1(self):
        """
        Test optimizing the population, no operations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.randomize_population = False
        config1.accept_new_best = True
        config1.num_of_mutations = 1
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulationNode6 = ESPopulationNode6(config1, ind1)

        while True:
            ind2: ESIndividual = population1.ps_process_data(ind1)
            if ind2.fitness < 1.0:
                break

        self.assertAlmostEqual(ind2.fitness, 0.0)
        self.assertEqual(ind2.data, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # type: ignore

        self.assertAlmostEqual(population1.population.es_get_best_fitness(), 0.0)

    def test_population_process_data2(self):
        """
        Test optimizing the population, three operations.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.randomize_population = False
        config1.accept_new_best = True
        config1.num_of_mutations = 1
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        config1.mutation_operations = [0, 1, 2]
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulationNode6 = ESPopulationNode6(config1, ind1)

        while True:
            ind2: ESIndividual = population1.ps_process_data(ind1)
            if ind2.fitness < 1.0:
                break

        self.assertAlmostEqual(ind2.fitness, 0.0)
        self.assertEqual(ind2.data, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # type: ignore

        self.assertAlmostEqual(population1.population.es_get_best_fitness(), 0.0)

        mut_counter: int = 0

        for ind in population1.population.population:
            if len(ind.mut_op_counter) > 1:
                mut_counter += 1

        self.assertGreater(mut_counter, 0)


if __name__ == "__main__":
    unittest.main()

