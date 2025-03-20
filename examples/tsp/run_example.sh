#!/usr/bin/env bash

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 tsp_main.py --server &
sleep 2
python3 tsp_main.py &
sleep 2
python3 tsp_main.py --reset_population --num_of_iterations 30000 &
sleep 2
python3 tsp_main.py --reset_population --num_of_iterations 40000 &
sleep 2
python3 tsp_main.py --reset_population --num_of_iterations 50000 &
