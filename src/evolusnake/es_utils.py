# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import fastrand


def uniform1() -> float:
    # Random float between -1.0 and 1.0
    return (fastrand.pcg32_uniform() * 2.0) - 1.0


def uniform2() -> float:
    # Random float between -0.01 and 0.01
    return (fastrand.pcg32_uniform() * 0.02) - 0.01


def uniform3() -> float:
    # Random float between 0.0001 and 0.1001
    return (fastrand.pcg32_uniform() * 0.1) + 0.0001


def shuffle_list(data: list):
    num_elems: int = len(data)

    for _ in range(num_elems):
        i: int = fastrand.pcg32bounded(num_elems)
        j: int = fastrand.pcg32bounded(num_elems)

        (data[i], data[j]) = (data[j], data[i])
