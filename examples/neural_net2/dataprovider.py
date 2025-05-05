# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


import random as rnd
from typing import Generator
import logging

logger = logging.getLogger(__name__)

class DataProvider:
    def __init__(self, data_values: list, batch_size: int):
        total_length = len(data_values)

        self.batch_size: int = batch_size
        self.training_data: list = []
        self.test_data: list = []

        rnd.seed()
        rnd.shuffle(data_values)

        # Use 80% for training and 20% for testing (20% is 1/5):
        test_limit: int = int(total_length / 5)

        for i in range(test_limit):
            self.test_data.append(data_values[i])

        for i in range(test_limit, total_length):
            self.training_data.append(data_values[i])

        self.training_size = len(self.training_data)
        self.test_size = len(self.test_data)

        logger.debug(f"batch size: {batch_size}")
        logger.debug(f"train size: {self.training_size}")
        logger.debug(f"test size: {self.test_size}")

    def get_batch(self) -> list:
        batch_idx: list = []

        for _ in range(self.batch_size):
            n = rnd.randrange(self.training_size)
            batch_idx.append(n)

        return batch_idx

    def test_batch(self) -> Generator[tuple[list, list], None, None]:
        for _ in range(self.batch_size):
            n = rnd.randrange(self.test_size)
            yield self.test_data[n]

