#!/bin/bash

#python main.py --num_users 5 --num_chosen_clients 5
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 10
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 30
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 50

python fed_dp.py --num_users 20 --num_chosen_clients 20 --local_bs 64 --privacy_budget 30 --num_rounds 50
