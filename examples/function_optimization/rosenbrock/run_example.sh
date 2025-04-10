#!/usr/bin/env bash

test_kind_1() {
    python3 rosenbrock_main.py --population_kind 1 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 1 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 1 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 1 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_2() {
    python3 rosenbrock_main.py --population_kind 2 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 2 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 2 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 2 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_3() {
    python3 rosenbrock_main.py --population_kind 3 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 3 --randomize_population --num_of_iterations 30000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 3 --randomize_population --num_of_iterations 40000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 3 --randomize_population --num_of_iterations 50000 &
    sleep 2
}

test_kind_4() {
    python3 rosenbrock_main.py --population_kind 4 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 4 --num_of_iterations 30000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 4 --num_of_iterations 40000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 4 --num_of_iterations 50000 &
    sleep 2
}

test_kind_5() {
    python3 rosenbrock_main.py --population_kind 5 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 5 --num_of_iterations 30000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 5 --num_of_iterations 40000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 5 --num_of_iterations 50000 &
    sleep 2
}

test_kind_6() {
    python3 rosenbrock_main.py --population_kind 6 --num_of_iterations 1000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 6 --randomize_population --num_of_iterations 1100 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 6 --randomize_population --num_of_iterations 1200 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 6 --randomize_population --num_of_iterations 1300 &
    sleep 2
}

test_kind_7() {
    python3 rosenbrock_main.py --population_kind 7 --num_of_iterations 1000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 7 --num_of_iterations 2000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 7 --num_of_iterations 3000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 7 --num_of_iterations 4000 &
    sleep 2
}

test_all_kinds() {
    python3 rosenbrock_main.py --population_kind 1 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 2 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 3 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 4 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 5 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 6 --num_of_iterations 1000 &
    sleep 2
    python3 rosenbrock_main.py --population_kind 7 --num_of_iterations 1000 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 rosenbrock_main.py --server &
sleep 2
test_kind_1
echo "All nodes running"
