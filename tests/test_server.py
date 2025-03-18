# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import unittest
import json
import os

# Local imports:
from evolusnake.es_config import ESConfiguration
#from evolusnake.es_population import ESPopulation

from evolusnake.es_server import ESServer
from tests.common import TestIndividual

# External imports:
from parasnake.ps_config import PSConfiguration


# TODO: ps_get_new_data, ps_process_result 


class TestServer(unittest.TestCase):
    def test_server_valid_config(self):
        """
        Test server init with a valid config.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)

        self.assertEqual(server1.population_size, config1.server_population_size)
        self.assertEqual(len(server1.population), config1.server_population_size)
        self.assertAlmostEqual(server1.target_fitness, config1.target_fitness)
        self.assertEqual(server1.result_filename, config1.result_filename)
        self.assertEqual(server1.save_new_fitness, config1.save_new_finess)
        self.assertEqual(server1.allow_same_fitness, config1.allow_same_fitness)
        self.assertEqual(server1.share_only_best, config1.share_only_best)
        self.assertEqual(server1.new_fitness_counter, 0)

    def test_server_invalid_config(self):
        """
        Test server init with an invalid config.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        config1.server_population_size = 0
        ind1: TestIndividual = TestIndividual()

        with self.assertRaises(ValueError):
            server1: ESServer = ESServer(config1, ind1)
            del server1

    def test_server_save_data1(self):
        """
        Test server saving data.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()
        ind1.fitness = 5.67
        ind1.data = [1, 2, 3, 4, 5]
        ind1.data_size = 5

        server1: ESServer = ESServer(config1, ind1)
        server1.population[0] = ind1

        filename = "test_data.json"
        server1.es_save_data(filename)

        with open(filename, "r") as f:
            data: dict = json.load(f)

        os.remove(filename)

        self.assertAlmostEqual(data["fitness"], ind1.fitness)
        self.assertEqual(data["data"], ind1.data)

    def test_server_is_job_done1(self):
        """
        Test if job is done.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        server1.target_fitness = 0.0
        server1.population[0].fitness = 0.0

        self.assertEqual(server1.ps_is_job_done(), True)
 
    def test_server_is_job_done2(self):
        """
        Test if job is done.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        server1.target_fitness = 0.0
        server1.population[0].fitness = 1.0

        self.assertEqual(server1.ps_is_job_done(), False)

    def test_server_get_new_data1(self):
        """
        Test generating new data for the node. Only best.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        config1.share_only_best = True
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        # TODO: implement

    def test_server_get_new_data2(self):
        """
        Test generating new data for the node. Choose randomly.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        config1.share_only_best = False
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        # TODO: implement

    def test_server_process_result(self):
        """
        Test receiving data from node.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        # TODO: implement

    def test_server_save_data2(self):
        """
        Test server saving data.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        ind1: TestIndividual = TestIndividual()
        ind1.fitness = 8.91
        ind1.data = [7, 8, 9, 10, 11]
        ind1.data_size = len(ind1.data)

        server1: ESServer = ESServer(config1, ind1)
        server1.population[0] = ind1

        filename = "test_data.json"
        server1.result_filename = filename
        server1.ps_save_data()

        with open(filename, "r") as f:
            data: dict = json.load(f)

        os.remove(filename)

        self.assertAlmostEqual(data["fitness"], ind1.fitness)
        self.assertEqual(data["data"], ind1.data)


if __name__ == "__main__":
    unittest.main()

