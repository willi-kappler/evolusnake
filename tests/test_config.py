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
            "target_fitness2": 3.5,
            "result_filename": "test_file1.json",
            "save_new_fitness": True,
            "allow_same_fitness": False,
            "share_only_best": False,
            "server_population_size": 55,
            "node_population_size": 77,
            "num_of_iterations": 12345,
            "num_of_mutations": 17,
            "accept_new_best": False,
            "randomize_population": True,
            "randomize_count": 3,
            "population_kind": 1,
            "mutation_operations": [7, 9, 11],
            "min_num_ind": 5,
            "sine_base": 12.6,
            "sine_amplitude": 33.4,
            "sine_frequency": 0.23,
            "limit_range": 1.23,
            "user_options": "some_options_1"
        }

        test_config_file: str = "test_config1.json"

        with open(test_config_file, "w") as f:
            json.dump(json_config, f)

        config1: ESConfiguration = ESConfiguration.from_json(test_config_file)
        os.remove(test_config_file)

        self.assertEqual(config1.server_mode, True)
        self.assertAlmostEqual(config1.target_fitness, 5.8)
        self.assertAlmostEqual(config1.target_fitness2, 3.5)
        self.assertEqual(config1.result_filename, "test_file1.json")
        self.assertEqual(config1.save_new_fitness, True)
        self.assertEqual(config1.allow_same_fitness, False)
        self.assertEqual(config1.share_only_best, False)
        self.assertEqual(config1.server_population_size, 55)
        self.assertEqual(config1.node_population_size, 77)
        self.assertEqual(config1.num_of_iterations, 12345)
        self.assertEqual(config1.num_of_mutations, 17)
        self.assertEqual(config1.accept_new_best, False)
        self.assertEqual(config1.randomize_population, True)
        self.assertEqual(config1.randomize_count, 3)
        self.assertEqual(config1.population_kind, 1)
        self.assertEqual(config1.mutation_operations, [7, 9, 11])
        self.assertEqual(config1.min_num_ind, 5)
        self.assertAlmostEqual(config1.sine_base, 12.6)
        self.assertAlmostEqual(config1.sine_amplitude, 33.4)
        self.assertAlmostEqual(config1.sine_frequency, 0.23)
        self.assertAlmostEqual(config1.limit_range, 1.23)
        self.assertEqual(config1.user_options, "some_options_1")

    def test_load_config2(self):
        """
        Test loading config from JSON.
        """

        json_config: dict = {
            "secret_key": "12345678901234567890123456789012",
            "server_mode": False,
            "target_fitness": 1.34,
            "target_fitness2": 8.29,
            "result_filename": "test_file2.json",
            "save_new_fitness": False,
            "allow_same_fitness": True,
            "share_only_best": True,
            "server_population_size": 88,
            "node_population_size": 99,
            "num_of_iterations": 33333,
            "num_of_mutations": 21,
            "accept_new_best": True,
            "randomize_population": False,
            "randomize_count": 7,
            "population_kind": 2,
            "mutation_operations": [3, 4, 10, 13],
            "min_num_ind": 7,
            "sine_base": 46.2,
            "sine_amplitude": 13.89,
            "sine_frequency": 0.043,
            "limit_range": 6.88,
            "user_options": "some_other_options_2"
        }

        test_config_file: str = "test_config2.json"

        with open(test_config_file, "w") as f:
            json.dump(json_config, f)

        config1: ESConfiguration = ESConfiguration.from_json(test_config_file)
        os.remove(test_config_file)

        self.assertEqual(config1.server_mode, False)
        self.assertAlmostEqual(config1.target_fitness, 1.34)
        self.assertAlmostEqual(config1.target_fitness2, 8.29)
        self.assertEqual(config1.result_filename, "test_file2.json")
        self.assertEqual(config1.save_new_fitness, False)
        self.assertEqual(config1.allow_same_fitness, True)
        self.assertEqual(config1.share_only_best, True)
        self.assertEqual(config1.server_population_size, 88)
        self.assertEqual(config1.node_population_size, 99)
        self.assertEqual(config1.num_of_iterations, 33333)
        self.assertEqual(config1.num_of_mutations, 21)
        self.assertEqual(config1.accept_new_best, True)
        self.assertEqual(config1.randomize_population, False)
        self.assertEqual(config1.randomize_count, 7)
        self.assertEqual(config1.population_kind, 2)
        self.assertEqual(config1.mutation_operations, [3, 4, 10, 13])
        self.assertEqual(config1.min_num_ind, 7)
        self.assertAlmostEqual(config1.sine_base, 46.2)
        self.assertAlmostEqual(config1.sine_amplitude, 13.89)
        self.assertAlmostEqual(config1.sine_frequency, 0.043)
        self.assertAlmostEqual(config1.limit_range, 6.88)
        self.assertEqual(config1.user_options, "some_other_options_2")


if __name__ == "__main__":
    unittest.main()

