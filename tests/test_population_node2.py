# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import unittest

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_population_node2 import ESPopulationNode2
from evolusnake.es_individual import ESIndividual

from tests.common import TestIndividual

# External imports:
from parasnake.ps_config import PSConfiguration


class TestPopulation(unittest.TestCase):
    def test_population_process_data(self):
        """
        Test optimizing the population.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.randomize_population = False
        config1.accept_new_best = True
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        population1: ESPopulationNode2 = ESPopulationNode2(config1, ind1)

        ind2: ESIndividual = population1.ps_process_data(ind1)

        self.assertAlmostEqual(ind2.fitness, 0.0)
        self.assertEqual(ind2.data, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # type: ignore

        self.assertAlmostEqual(population1.population.best_fitness, 0.0)


if __name__ == "__main__":
    unittest.main()

