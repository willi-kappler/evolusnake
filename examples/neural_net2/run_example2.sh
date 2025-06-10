#!/usr/bin/env bash

main_file="neural_net_main.py"

run_kind() {
    echo "Run kind $1."

    num_of_mutations="1"

    case $1 in
    6)
        num_of_mutations="20"
    esac

    python3 $main_file -k $1 -m $num_of_mutations -i 3000 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 3200 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 3400 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 3600 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 3800 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4000 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4200 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4400 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4600 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 4800 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 5000 --user_options net=1 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i 5200 --user_options net=1 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 $main_file --server &
sleep 2
run_kind 1
echo "All nodes running"
