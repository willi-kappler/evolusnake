#!/usr/bin/env bash

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 tsp_main.py --server &
sleep 2
python3 tsp_main.py --num_of_mutations 1 &
sleep 2
python3 tsp_main.py --num_of_mutations 10 &
sleep 2
python3 tsp_main.py --reset_population --num_of_mutations 1 &
sleep 2
python3 tsp_main.py --reset_population --num_of_mutations 10 &
