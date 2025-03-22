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
        self.population_kind: int = 1
        self.increase_iteration: int = 0
        self.increase_mutation: int = 0

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
                case "result_filename":
                    config.result_filename = value
                case "save_new_fitness":
                    config.save_new_fitness = value
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
                case "population_kind":
                    config.population_kind = value
                case "increase_iteration":
                    config.increase_iteration = value
                case "increase_mutation":
                    config.increase_mutation = value
                case _:
                    logger.debug(f"Unknown evolusnake configuration option: {key=}, {value=}")

        return config

    def from_command_line(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--server", action="store_true")
        parser.add_argument("--target_fitness", type=float)
        parser.add_argument("--population_size", type=int)
        parser.add_argument("--num_of_mutations", type=int)
        parser.add_argument("--num_of_iterations", type=int)
        parser.add_argument("--population_kind", type=int)
        parser.add_argument("--randomize_population", action="store_true")
        parser.add_argument("--increase_iteration", type=int)
        parser.add_argument("--increase_mutation", type=int)

        args = parser.parse_args()

        self.server_mode = args.server

        if args.target_fitness != None:
            self.target_fitness = args.target_fitness

        if args.population_size != None:
            self.node_population_size = args.population_size

        if args.num_of_mutations != None:
            self.num_of_mutations = args.num_of_mutations

        if args.num_of_iterations != None:
            self.num_of_iterations = args.num_of_iterations

        if args.population_kind != None:
            self.population_kind = args.population_kind

        self.randomize_population = args.randomize_population

        if args.increase_iteration != None:
            self.increase_iteration = args.increase_iteration

        if args.increase_mutation != None:
            self.increase_mutation = args.increase_mutation

        # print(f"{self.server_mode}")
        # print(f"{self.target_fitness}")
        # print(f"{self.node_population_size}")
        # print(f"{self.num_of_mutations}")
        # print(f"{self.num_of_iterations}")
        # print(f"{self.population_kind}")


