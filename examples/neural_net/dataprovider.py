# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


import random as rnd
from typing import Generator

class DataProvider:
    def __init__(self, input_values: list, expected_output: list, batch_size: int):
        total_length = len(input_values)

        # assert length of input_values and expected_output

        self.batch_size: int = batch_size

        self.training_data: list = []
        self.training_results: list = []

        self.test_data: list = []
        self.test_results: list = []

        for i in range(total_length):
            # Use 80% for training and 20% for testing (20% is 1/5):
            n = rnd.randrange(5)
            if n == 0:
                self.test_data.append(input_values[i])
                self.test_results.append(expected_output[i])
            else:
                self.training_data.append(input_values[i])
                self.training_results.append(expected_output[i])

        self.training_size = len(self.training_data)
        self.test_size = len(self.test_data)

    def training_batch(self) -> Generator[tuple[list, list], None, None]:
        for _ in range(self.batch_size):
            n = rnd.randrange(self.training_size)
            data: list = self.training_data[n]
            result: list = self.training_results[n]
            yield (data, result)

    def test_batch(self) -> Generator[tuple[list, list], None, None]:
        for _ in range(self.batch_size):
            n = rnd.randrange(self.test_size)
            data: list = self.test_data[n]
            result: list = self.test_results[n]
            yield (data, result)

