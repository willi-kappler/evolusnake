# This file is part of Evolusnake, evolutionary algorithms in Python
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

"""
This module defines the configuration.
"""

# Python std lib:
import json
import logging
import argparse
from typing import Any

# External imports:
from parasnake.ps_config import PSConfiguration

logger = logging.getLogger(__name__)


class ESConfiguration:
    """
    This class contains all the configuration options.
    """

    def __init__(self):
        self.parasnake_config: PSConfiguration = PSConfiguration("12345678901234567890123456789012")

        # Server config:
        self.server_mode: bool = False
        self.target_fitness: float = 0.0
        self.target_fitness2: float = 0.0
        self.result_filename: str = "best_result.json"
        self.save_new_fitness: bool = False
        self.allow_same_fitness: bool = False
        self.share_only_best: bool = False
        self.server_population_size: int = 10

        # Node config:
        self.node_population_size: int = 10
        self.num_of_iterations: int = 1000
        self.num_of_mutations: int = 10
        self.accept_new_best: bool = True
        self.randomize_population: bool = False
        self.randomize_count: int = 5
        self.population_kind: int = 1
        self.mutation_operations: list = [0]
        self.min_num_ind: int = 2
        self.sine_base: float = 100.0
        self.sine_amplitude: float = 50.0
        self.sine_frequency: float = 0.01
        self.limit_range: float = 5.0

        # User defined options:
        self.user_options: str = ""

    @staticmethod
    def from_json(file_name) -> Any:
        """
        Load the configuration (JSON format) from the given file name.

        :param file_name: File name of the configuration.
        :return: A valid configuration from the given JSON file.
        """

        logger.debug(f"Load configuration from file: {file_name}.")

        with open(file_name, "r") as f:
            data = json.load(f)

        config = ESConfiguration()

        # Load config data for Parasnake:
        config.parasnake_config = PSConfiguration.from_json(file_name)

        for key, value in data.items():
            match key:
                case "server_mode":
                    config.server_mode = value
                case "target_fitness":
                    config.target_fitness = value
                case "target_fitness2":
                    config.target_fitness2 = value
                case "result_filename":
                    config.result_filename = value
                case "save_new_fitness":
                    config.save_new_fitness = value
                case "allow_same_fitness":
                    config.allow_same_fitness = value
                case "share_only_best":
                    config.share_only_best = value
                case "server_population_size":
                    config.server_population_size = value
                case "node_population_size":
                    config.node_population_size = value
                case "num_of_iterations":
                    config.num_of_iterations = value
                case "num_of_mutations":
                    config.num_of_mutations = value
                case "accept_new_best":
                    config.accept_new_best = value
                case "randomize_population":
                    config.randomize_population = value
                case "randomize_count":
                    config.randomize_count = value
                case "population_kind":
                    config.population_kind = value
                case "mutation_operations":
                    config.mutation_operations = value
                case "min_num_ind":
                    config.min_num_ind = value
                case "sine_base":
                    config.sine_base = value
                case "sine_amplitude":
                    config.sine_amplitude = value
                case "sine_frequency":
                    config.sine_frequency = value
                case "limit_range":
                    config.limit_range = value
                case "user_options":
                    config.user_options = value
                case _:
                    logger.debug(f"Unknown evolusnake configuration option: {key=}, {value=}")

        return config

    def from_command_line(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-s", "--server", action="store_true")
        parser.add_argument("-f", "--target_fitness", type=float)
        parser.add_argument("-f2", "--target_fitness2", type=float)
        parser.add_argument("-p", "--population_size", type=int)
        parser.add_argument("-m", "--num_of_mutations", type=int)
        parser.add_argument("-i", "--num_of_iterations", type=int)
        parser.add_argument("-k", "--population_kind", type=int)
        parser.add_argument("-r", "--randomize_population", action="store_true")
        parser.add_argument("-o", "--mutation_operations")
        parser.add_argument("--randomize_count", type=int)
        parser.add_argument("--min_num_ind", type=int)
        parser.add_argument("--sine_base", type=float)
        parser.add_argument("--sine_amplitude", type=float)
        parser.add_argument("--sine_frequency", type=float)
        parser.add_argument("--limit_range", type=float)
        parser.add_argument("--user_options")

        args = parser.parse_args()

        self.server_mode = args.server

        if args.target_fitness is not None:
            self.target_fitness = args.target_fitness

        if args.target_fitness2 is not None:
            self.target_fitness2 = args.target_fitness2

        if args.population_size is not None:
            self.node_population_size = args.population_size

        if args.num_of_mutations is not None:
            self.num_of_mutations = args.num_of_mutations

        if args.num_of_iterations is not None:
            self.num_of_iterations = args.num_of_iterations

        if args.population_kind is not None:
            self.population_kind = args.population_kind

        self.randomize_population = args.randomize_population

        if args.randomize_count is not None:
            self.randomize_count = args.randomize_count

        if args.mutation_operations is not None:
            self.mutation_operations = [int(n) for n in args.mutation_operations.split(",")]

        if args.min_num_ind is not None:
            self.min_num_ind = args.min_num_ind

        if args.sine_base is not None:
            self.sine_base = args.sine_base

        if args.sine_amplitude is not None:
            self.sine_amplitude = args.sine_amplitude

        if args.sine_frequency is not None:
            self.sine_frequency = args.sine_frequency

        if args.limit_range is not None:
            self.limit_range = args.limit_range

        if args.user_options is not None:
            self.user_options = args.user_options
