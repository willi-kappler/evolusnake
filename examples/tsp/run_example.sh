#!/usr/bin/env bash

test_kind_1() {
    python3 tsp_main.py --population_kind 1 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 1 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_2() {
    python3 tsp_main.py --population_kind 2 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 2 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_3() {
    python3 tsp_main.py --population_kind 3 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 3 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_4() {
    python3 tsp_main.py --population_kind 4 --num_of_iterations 10000 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 20000 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 30000 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 4 --num_of_iterations 40000 --num_of_mutations 1 &
    sleep 2
}

test_kind_5() {
    python3 tsp_main.py --population_kind 5 &
    sleep 2
    python3 tsp_main.py --population_kind 5 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 5 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 5 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_6() {
    python3 tsp_main.py --population_kind 6 --num_of_iterations 1000 &
    sleep 2
    python3 tsp_main.py --population_kind 6 --randomize_population --num_of_iterations 1100 &
    sleep 2
    python3 tsp_main.py --population_kind 6 --randomize_population --num_of_iterations 1200 &
    sleep 2
    python3 tsp_main.py --population_kind 6 --randomize_population --num_of_iterations 1300 &
    sleep 2
}

test_kind_7() {
    python3 tsp_main.py --population_kind 7 --num_of_iterations 1000 &
    sleep 2
    python3 tsp_main.py --population_kind 7 --num_of_iterations 2000 &
    sleep 2
    python3 tsp_main.py --population_kind 7 --num_of_iterations 3000 &
    sleep 2
    python3 tsp_main.py --population_kind 7 --num_of_iterations 4000 &
    sleep 2
}

test_kind_8() {
    python3 tsp_main.py --population_kind 8 &
    sleep 2
    python3 tsp_main.py --population_kind 8 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 8 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 8 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_9() {
    python3 tsp_main.py --population_kind 9 &
    sleep 2
    python3 tsp_main.py --population_kind 9 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 9 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 9 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_10() {
    python3 tsp_main.py --population_kind 10 &
    sleep 2
    python3 tsp_main.py --population_kind 10 --num_of_mutations 1 &
    sleep 2
    python3 tsp_main.py --population_kind 10 --randomize_population --num_of_iterations 50000 &
    sleep 2
    python3 tsp_main.py --population_kind 10 --randomize_population --num_of_iterations 50000 --num_of_mutations 1 &
    sleep 2
}

test_kind_11() {
    python3 tsp_main.py --population_kind 11 --num_of_mutations 1 --num_of_iterations 10000 &
    sleep 2
    python3 tsp_main.py --population_kind 11 --num_of_mutations 1 --num_of_iterations 20000 &
    sleep 2
    python3 tsp_main.py --population_kind 11 --num_of_mutations 1 --num_of_iterations 30000 &
    sleep 2
    python3 tsp_main.py --population_kind 11 --num_of_mutations 1 --num_of_iterations 40000 &
    sleep 2
}

test_all_kinds() {
    python3 tsp_main.py --population_kind 1 &
    sleep 2
    python3 tsp_main.py --population_kind 2 &
    sleep 2
    python3 tsp_main.py --population_kind 3 &
    sleep 2
    python3 tsp_main.py --population_kind 4 &
    sleep 2
    python3 tsp_main.py --population_kind 5 &
    sleep 2
    python3 tsp_main.py --population_kind 6 --num_of_iterations 1000 &
    sleep 2
    python3 tsp_main.py --population_kind 7 --num_of_iterations 1000 &
    sleep 2
    python3 tsp_main.py --population_kind 8 &
    sleep 2
    python3 tsp_main.py --population_kind 9 &
    sleep 2
    python3 tsp_main.py --population_kind 10 &
    sleep 2
    python3 tsp_main.py --population_kind 11 --sine_base 10000.0 --sine_amplitude 2000.0 --sine_freq 0.0001 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 tsp_main.py --server &
sleep 2
test_kind_1
echo "All nodes running"
