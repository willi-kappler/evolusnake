#!/usr/bin/env bash

main_file="neural_net_main.py"

run_kind() {
    echo "Run kind $1."

    num_of_mutations="1"
    randomize="-r"
    iterations1="10000"
    iterations2="20000"
    iterations3="30000"
    iterations4="40000"

    case $1 in
    4 | 11)
        randomize=""
        ;;
    6)
        num_of_mutations="20"
        iterations1="1000"
        iterations2="2000"
        iterations3="3000"
        iterations4="4000"
        ;;
    7)
        randomize=""
        iterations1="1000"
        iterations2="2000"
        iterations3="3000"
        iterations4="4000"
        ;;
    esac

    python3 $main_file -k $1 -m $num_of_mutations -i $iterations1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations2 $randomize &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations3 $randomize &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations4 $randomize &
    sleep 2
}

run_all_kinds() {
    echo "Run all population kinds."

    python3 $main_file -k 1 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 2 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 3 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 4 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 5 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 6 -m 20 -i 1000 &
    sleep 2
    python3 $main_file -k 7 -m 1 -i 1000 &
    sleep 2
    python3 $main_file -k 8 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 9 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 10 -m 1 -i 10000 &
    sleep 2
    python3 $main_file -k 11 -m 1 -i 10000 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 $main_file --server &
sleep 2
run_kind 1
echo "All nodes running"
