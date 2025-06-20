# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import logging
import pathlib
from typing import override, Self

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer
import evolusnake.es_utils as utils


logger = logging.getLogger(__name__)


class SudokuIndividual(ESIndividual):
    def __init__(self):
        super().__init__()

        # Solution:
        # 7, 8, 1,   3, 9, 4,   6, 2, 5,
        # 2, 6, 3,   5, 1, 7,   9, 4, 8,
        # 4, 9, 5,   6, 8, 2,   1, 7, 3,
        #
        # 1, 3, 8,   4, 5, 6,   7, 9, 2,
        # 9, 2, 4,   7, 3, 1,   8, 5, 6,
        # 5, 7, 6,   8, 2, 9,   3, 1, 4,
        #
        # 6, 5, 7,   9, 4, 3,   2, 8, 1,
        # 3, 4, 2,   1, 7, 8,   5, 6, 9,
        # 8, 1, 9,   2, 6, 5,   4, 3, 7

        self.numbers1: list = [
            0, 8, 0,   0, 9, 4,   0, 0, 0,
            2, 0, 3,   0, 0, 0,   9, 4, 0,
            0, 0, 0,   0, 0, 2,   1, 0, 3,

            0, 0, 8,   0, 0, 0,   7, 9, 0,
            9, 2, 0,   0, 0, 0,   0, 5, 6,
            0, 7, 6,   0, 0, 0,   3, 0, 0,

            0, 5, 7,   0, 0, 0,   2, 0, 1,
            3, 0, 2,   1, 0, 0,   0, 0, 0,
            0, 0, 0,   2, 6, 0,   0, 3, 0
        ]

        self.numbers2: list = self.numbers1[:]

        self.empty_positions: list = []

        for c in range(9):
            for r in range(9):
                if self.get_value1(c, r) == 0:
                    self.empty_positions.append((c, r))

    def get_value1(self, col: int, row: int) -> int:
        return self.numbers1[(row * 9) + col]

    def get_value2(self, col: int, row: int) -> int:
        return self.numbers2[(row * 9) + col]

    def get_value3(self, col: int, row: int) -> int:
        if col == -1:
            col = 8
        elif col == 9:
            col = 0

        if row == -1:
            row = 8
        elif row == 9:
            row = 0

        return self.get_value2(col, row)

    def set_value2(self, col: int, row: int, val: int):
        self.numbers2[(row * 9) + col] = val

    def check_pos(self, col: int, row: int, in_use: set[int]) -> int:
        n: int = self.get_value2(col, row)

        if (n == 0) or (n in in_use):
            return 1
        else:
            in_use.add(n)
            return 0

    def check_line(self, col: int, row: int, col_inc: int, row_inc: int) -> int:
        errors: int = 0
        in_use: set[int] = set()
        c: int = col
        r: int = row

        for _ in range(9):
            errors += self.check_pos(c, r, in_use)
            c += col_inc
            r += row_inc

        return errors

    def check_row(self, row: int) -> int:
        return self.check_line(0, row, 1, 0)

    def check_col(self, col: int) -> int:
        return self.check_line(col, 0, 0, 1)

    def check_block(self, i: int, j: int) -> int:
        errors: int = 0
        in_use: set[int] = set()

        for u in range(3):
            for v in range(3):
                errors += self.check_pos(i + u, j + v, in_use)

        return errors

    def fill_numbers(self, possible_numbers, positions):
        possible_numbers2: list = list(possible_numbers)
        utils.es_shuffle_list(possible_numbers2)

        for (c, r) in positions:
            self.set_value2(c, r, possible_numbers2.pop())

    def set_random_number(self):
        (c, r) = utils.es_choice(self.empty_positions)
        n = utils.es_rand_int(9) + 1
        self.set_value2(c, r, n)

    def swap_two_numbers(self):
        (c1, r1) = utils.es_choice(self.empty_positions)
        (c2, r2) = utils.es_choice(self.empty_positions)

        n1 = self.get_value2(c1, r1)
        n2 = self.get_value2(c2, r2)

        self.set_value2(c1, r1, n2)
        self.set_value2(c2, r2, n1)

    def random_cross(self):
        (c, r) = utils.es_choice(self.empty_positions)
        possible_numbers: set = {1, 2, 3, 4, 5, 6, 7, 8, 9}

        n: int = self.get_value3(c - 1, r)
        possible_numbers.discard(n)

        n: int = self.get_value3(c + 1, r)
        possible_numbers.discard(n)

        n: int = self.get_value3(c, r - 1)
        possible_numbers.discard(n)

        n: int = self.get_value3(c, r + 1)
        possible_numbers.discard(n)

        possible_numbers2: list = list(possible_numbers)
        i = utils.es_rand_int(5)
        n = possible_numbers2[i]

        self.set_value2(c, r, n)

    def random_col(self):
        (c, _) = utils.es_choice(self.empty_positions)
        possible_numbers: set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        new_pos: list = []

        for r in range(9):
            n: int = self.get_value1(c, r) > 0
            if n > 0:
                possible_numbers.discard(n)
            else:
                new_pos.append((c, r))

        self.fill_numbers(possible_numbers, new_pos)

    def random_row(self):
        (_, r) = utils.es_choice(self.empty_positions)
        possible_numbers: set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        new_pos: list = []

        for c in range(9):
            n: int = self.get_value1(c, r) > 0
            if n > 0:
                possible_numbers.discard(n)
            else:
                new_pos.append((c, r))

        self.fill_numbers(possible_numbers, new_pos)

    def random_block(self):
        (c1, r1) = utils.es_choice(self.empty_positions)
        possible_numbers: set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        new_pos: list = []

        bc: int = int(c1 / 3)
        br: int = int(r1 / 3)

        for c2 in range(3):
            for r2 in range(3):
                c: int = (bc * 3) + c2
                r: int = (br * 3) + r2

                n = self.get_value1(c, r)
                if n > 0:
                    possible_numbers.discard(n)
                else:
                    new_pos.append((c, r))

        self.fill_numbers(possible_numbers, new_pos)

    @override
    def es_mutate(self, mut_op: int):
        match mut_op:
            case 0:
                self.set_random_number()
            case 1:
                self.swap_two_numbers()
            case 2:
                self.random_cross()
            case 3:
                self.random_col()
            case 4:
                self.random_row()
            case 5:
                self.random_block()

    @override
    def es_randomize(self):
        for (c, r) in self.empty_positions:
            n = utils.es_rand_int(9) + 1
            self.set_value2(c, r, n)

    @override
    def es_calculate_fitness(self):
        errors: int = 0

        # Check rows:
        for row in range(9):
            errors += self.check_row(row)

        # Check cols:
        for col in range(9):
            errors += self.check_col(col)

        # Check all blocks:
        for i in range(0, 7, 3):
            for j in range(0, 7, 3):
                errors += self.check_block(i, j)

        self.fitness = float(errors)

    @override
    def es_clone(self) -> Self:
        new = SudokuIndividual()
        new.numbers2 = self.numbers2[:]

        return new  # type: ignore

    @override
    def es_from_server(self, other):
        self.numbers2 = other.numbers2

    @override
    def es_to_json(self) -> dict:
        data = {
            "numbers2": self.numbers2
        }

        return data

    @override
    def es_from_json(self, data: dict):
        self.numbers2 = data["numbers2"]


def main():
    config = ESConfiguration.from_json("sudoku2_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "sudoku2_server.log"

    if not server_mode:
        sudoku_num: int = 1

        while True:
            log_file_name = f"sudoku2_node_{sudoku_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                sudoku_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    ind = SudokuIndividual()

    config.target_fitness = 0.0

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
    else:
        print("Create and start node.")
        population = es_select_population(config, ind)
        population.ps_run()


if __name__ == "__main__":
    main()

