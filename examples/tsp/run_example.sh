#!/usr/bin/env bash

reset

export PYTHONPATH=$PYTHONPATH:"../../src/"

python3 tsp_main.py --server &
sleep 2
python3 tsp_main.py &
sleep 2
python3 tsp_main.py --reset_population &
sleep 2
python3 tsp_main.py --population_kind 2 &
sleep 2
python3 tsp_main.py --population_kind 3 &
