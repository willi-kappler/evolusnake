# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake


# Python std lib:
import time

# External imports:
import fastrand


def es_init_seed():
    current_time: float = time.time()
    fastrand.pcg32_seed(int(current_time))


def es_uniform1() -> float:
    # Random float between -1.0 and 1.0
    return (fastrand.pcg32_uniform() * 2.0) - 1.0


def es_uniform2() -> float:
    # Random float between -0.01 and 0.01
    return (fastrand.pcg32_uniform() * 0.02) - 0.01


def es_uniform3() -> float:
    # Random float between 0.0001 and 0.1001
    return (fastrand.pcg32_uniform() * 0.1) + 0.0001


def es_uniform4() -> float:
    # Random float between 0.0 and 1.0
    return fastrand.pcg32_uniform()


def es_uniform5(lower: float, upper: float) -> float:
    # Random float between lower and upper
    diff: float = upper - lower
    return (fastrand.pcg32_uniform() * diff) + lower


def es_rand_int(limit: int) -> int:
    return fastrand.pcg32bounded(limit)


def es_shuffle_list(data: list):
    num_elems: int = len(data)

    for _ in range(num_elems):
        i: int = fastrand.pcg32bounded(num_elems)
        j: int = fastrand.pcg32bounded(num_elems)

        (data[i], data[j]) = (data[j], data[i])


def es_random_swap(data: list):
    num_elems: int = len(data)

    i: int = fastrand.pcg32bounded(num_elems)
    j: int = fastrand.pcg32bounded(num_elems)

    while i == j:
        j = fastrand.pcg32bounded(num_elems)

    (data[i], data[j]) = (data[j], data[i])


def es_choice(data: list):
    num_elems: int = len(data)

    i: int = fastrand.pcg32bounded(num_elems)

    return data[i]
