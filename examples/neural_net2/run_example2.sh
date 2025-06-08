#!/usr/bin/env bash

main_file="neural_net_main.py"

run_kind() {
    echo "Run kind $1."

    num_of_mutations="1"

    case $1 in
    6)
        num_of_mutations="20"
    esac

    python3 $main_file -k $1 -m $num_of_mutations -i 3000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 3500 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4500 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 5000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 5500 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 6000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 6500 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 7000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 7500 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 8000 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 8500 --user_options net=6 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 $main_file --server &
sleep 2
run_kind 1
echo "All nodes running"
