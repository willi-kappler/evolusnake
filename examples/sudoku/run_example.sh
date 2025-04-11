#!/usr/bin/env bash

test_kind_1() {
    python3 sudoku_main.py --population_kind 1 &
    sleep 2
    python3 sudoku_main.py --population_kind 1 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 1 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 1 --randomize_population &
    sleep 2
}

test_kind_2() {
    python3 sudoku_main.py --population_kind 2 &
    sleep 2
    python3 sudoku_main.py --population_kind 2 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 2 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 2 --randomize_population &
    sleep 2
}

test_kind_3() {
    python3 sudoku_main.py --population_kind 3 &
    sleep 2
    python3 sudoku_main.py --population_kind 3 --randomize_population &
    sleep
    python3 sudoku_main.py --population_kind 3 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 3 --randomize_population &
    sleep 2
}

test_kind_4() {
    python3 sudoku_main.py --population_kind 4 &
    sleep 2
    python3 sudoku_main.py --population_kind 4 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 4 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 4 --randomize_population &
    sleep 2
}

test_kind_5() {
    python3 sudoku_main.py --population_kind 5 &
    sleep 2
    python3 sudoku_main.py --population_kind 5 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 5 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 5 --randomize_population &
    sleep 2
}

test_kind_6() {
    python3 sudoku_main.py --population_kind 6 &
    sleep 2
    python3 sudoku_main.py --population_kind 6 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 6 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 6 --randomize_population &
    sleep 2
}

test_kind_7() {
    python3 sudoku_main.py --population_kind 7 &
    sleep 2
    python3 sudoku_main.py --population_kind 7 &
    sleep 2
    python3 sudoku_main.py --population_kind 7 &
    sleep 2
    python3 sudoku_main.py --population_kind 7 &
    sleep 2
}

test_kind_8() {
    python3 sudoku_main.py --population_kind 8 &
    sleep 2
    python3 sudoku_main.py --population_kind 8 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 8 --randomize_population &
    sleep 2
    python3 sudoku_main.py --population_kind 8 --randomize_population &
    sleep 2
}

test_all_kinds() {
    python3 sudoku_main.py --population_kind 1 &
    sleep 2
    python3 sudoku_main.py --population_kind 2 &
    sleep 2
    python3 sudoku_main.py --population_kind 3 &
    sleep 2
    python3 sudoku_main.py --population_kind 4 &
    sleep 2
    python3 sudoku_main.py --population_kind 5 &
    sleep 2
    python3 sudoku_main.py --population_kind 6 &
    sleep 2
    python3 sudoku_main.py --population_kind 7 &
    sleep 2
    python3 sudoku_main.py --population_kind 8 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 sudoku_main.py --server &
sleep 2
test_kind_1
echo "All nodes running"
