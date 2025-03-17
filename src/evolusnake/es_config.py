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
        self.save_new_finess: bool = False
        self.allow_same_fitness: bool = False
        self.share_only_best: bool = False
        self.server_population_size: int = 10

        # Node config:
        self.node_population_size: int = 10
        self.num_of_iterations: int = 1000
        self.num_of_mutations: int = 10
        self.accept_new_best: bool = True
        self.reset_population: bool = False
        self.population_kind: int = 1

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
                case "save_new_finess":
                    config.save_new_finess = value
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
                case "population_kind":
                    config.population_kind = value
                case _:
                    raise ValueError(f"Unknown configuration option: {key=}, {value=}")

        return config

