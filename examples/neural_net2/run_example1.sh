#!/usr/bin/env bash

main_file="neural_net_main.py"

run_kind() {
    echo "Run kind $1."

    num_of_mutations="1"
    iterations1="3000"

    case $1 in
    6)
        num_of_mutations="20"
        ;;
    esac

    # python3 -m cProfile $main_file -k $1 -m $num_of_mutations -i $iterations1 --user_options net=6 > profile_out.txt &
    # sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations1 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations1 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations1 --user_options net=6 &
    sleep 2
    python3 $main_file -k $1 -m $num_of_mutations -i $iterations1 --user_options net=6 &
    sleep 2
}

run_all_nets() {
    echo "Run all nets."

    iterations1="3000"

    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=1 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=2 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=3 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=4 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=5 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=6 &
    sleep 2
    python3 $main_file -k 1 -m 1 -i $iterations1 --user_options net=7 &
    sleep 2
}

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 $main_file --server &
sleep 2
run_kind 1
#run_all_nets
echo "All nodes running"
