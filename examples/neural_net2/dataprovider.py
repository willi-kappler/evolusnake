# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


# Python std lib:
from typing import Generator, override
import logging

# Local imports
from evolusnake.es_population import ESPopulation, ESIterationCallBack
import evolusnake.es_utils as utils


logger = logging.getLogger(__name__)


class IterationNeural(ESIterationCallBack):
    def __init__(self):
        pass

    @override
    def es_fraction_iteration(self, population: ESPopulation):
        population.population[0].data_provider.create_batch_indices()  # type: ignore

        for ind in population.population:
            ind.es_calculate_fitness()

    @override
    def es_get_iteration_factor(self) -> int:
        return 3


class DataProvider:
    def __init__(self, data_values: list, batch_size: int):
        total_length = len(data_values)

        self.batch_size: int = batch_size
        self.training_data: list = []
        self.test_data: list = []

        # Init random number generator:
        utils.es_init_seed()
        utils.es_shuffle_list(data_values)

        # Use 80% for training and 20% for testing (20% is 1/5):
        test_limit: int = int(total_length / 5)

        for i in range(test_limit):
            self.test_data.append(data_values[i])

        for i in range(test_limit, total_length):
            self.training_data.append(data_values[i])

        self.training_size = len(self.training_data)
        self.test_size = len(self.test_data)

        self.batch_indices: list = [0] * batch_size
        self.create_batch_indices()

        logger.debug(f"batch size: {batch_size}")
        logger.debug(f"train size: {self.training_size}")
        logger.debug(f"test size: {self.test_size}")

    def create_batch_indices(self):
        for i in range(self.batch_size):
            self.batch_indices[i] = utils.es_rand_int(self.training_size)

    def training_batch(self) -> Generator[tuple[list, list], None, None]:
        for i in self.batch_indices:
            yield self.training_data[i]

    def test_batch(self) -> Generator[tuple[list, list], None, None]:
        for _ in range(self.batch_size):
            n = utils.es_rand_int(self.test_size)
            yield self.test_data[n]
