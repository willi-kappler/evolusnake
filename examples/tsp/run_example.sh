#!/usr/bin/env bash

test_kind_1() {
    python3 tsp_main.py --population_kind 1 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_2() {
    python3 tsp_main.py --population_kind 2 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_3() {
    python3 tsp_main.py --population_kind 3 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_4() {
    python3 tsp_main.py --population_kind 4 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 30000 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 40000 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 50000 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 tsp_main.py --server &
sleep 2
test_kind_1
echo "All nodes running"
