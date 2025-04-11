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


class TestConfiguration(unittest.TestCase):
    def test_load_config1(self):
        """
        Test loading config from JSON.
        """

        json_config: dict = {
            "secret_key": "12345678901234567890123456789012",
            "server_mode": True,
            "target_fitness": 5.8,
            "result_filename": "test_file1.json",
            "save_new_fitness": True,
            "share_only_best": False,
            "server_population_size": 55,
            "node_population_size": 77,
            "num_of_iterations": 12345,
            "num_of_mutations": 17,
            "accept_new_best": False,
            "population_kind": 1,
            "increase_iteration": 5,
            "increase_mutation": 8,
            "mutation_operations": [7, 9, 11]
        }

        test_config_file: str = "test_config1.json"

        with open(test_config_file, "w") as f:
            json.dump(json_config, f)

        config1: ESConfiguration = ESConfiguration.from_json(test_config_file)
        os.remove(test_config_file)

        self.assertEqual(config1.server_mode, True)
        self.assertAlmostEqual(config1.target_fitness, 5.8)
        self.assertEqual(config1.result_filename, "test_file1.json")
        self.assertEqual(config1.save_new_fitness, True)
        self.assertEqual(config1.share_only_best, False)
        self.assertEqual(config1.server_population_size, 55)
        self.assertEqual(config1.node_population_size, 77)
        self.assertEqual(config1.num_of_iterations, 12345)
        self.assertEqual(config1.num_of_mutations, 17)
        self.assertEqual(config1.accept_new_best, False)
        self.assertEqual(config1.population_kind, 1)
        self.assertEqual(config1.increase_iteration, 5)
        self.assertEqual(config1.increase_mutation, 8)
        self.assertEqual(config1.mutation_operations, [7, 9, 11])

    def test_load_config2(self):
        """
        Test loading config from JSON.
        """

        json_config: dict = {
            "secret_key": "12345678901234567890123456789012",
            "server_mode": False,
            "target_fitness": 1.34,
            "result_filename": "test_file2.json",
            "save_new_fitness": False,
            "share_only_best": True,
            "server_population_size": 88,
            "node_population_size": 99,
            "num_of_iterations": 33333,
            "num_of_mutations": 21,
            "accept_new_best": True,
            "population_kind": 2,
            "increase_iteration": 11,
            "increase_mutation": 19,
            "mutation_operations": [3, 4, 10, 13]
        }

        test_config_file: str = "test_config2.json"

        with open(test_config_file, "w") as f:
            json.dump(json_config, f)

        config1: ESConfiguration = ESConfiguration.from_json(test_config_file)
        os.remove(test_config_file)

        self.assertEqual(config1.server_mode, False)
        self.assertAlmostEqual(config1.target_fitness, 1.34)
        self.assertEqual(config1.result_filename, "test_file2.json")
        self.assertEqual(config1.save_new_fitness, False)
        self.assertEqual(config1.share_only_best, True)
        self.assertEqual(config1.server_population_size, 88)
        self.assertEqual(config1.node_population_size, 99)
        self.assertEqual(config1.num_of_iterations, 33333)
        self.assertEqual(config1.num_of_mutations, 21)
        self.assertEqual(config1.accept_new_best, True)
        self.assertEqual(config1.population_kind, 2)
        self.assertEqual(config1.increase_iteration, 11)
        self.assertEqual(config1.increase_mutation, 19)
        self.assertEqual(config1.mutation_operations, [3, 4, 10, 13])


if __name__ == "__main__":
    unittest.main()

