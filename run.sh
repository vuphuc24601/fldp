#!/bin/bash

#python main.py --num_users 5 --num_chosen_clients 5
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 10
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 30
#python fed_dp.py --num_users 5 --num_chosen_clients 5 --privacy_budget 50

python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --privacy_budget 10
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 512
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64

python3 main.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 32

python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 32 --privacy_budget 10
python3 fed_dp.py --num_users 5 --num_chosen_clients 5 --num_data_per_user 512 --local_bs 64 --privacy_budget 30
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --privacy_budget 10 --num_rounds 50

# aggregation times
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 10
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 8
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 6
python3 fed_dp.py --num_users 20 --num_chosen_clients 20 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 4

python3 fed_dp.py --num_users 20 --num_chosen_clients 10 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 10
python3 fed_dp.py --num_users 20 --num_chosen_clients 10 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 8
python3 fed_dp.py --num_users 20 --num_chosen_clients 10 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 6
python3 fed_dp.py --num_users 20 --num_chosen_clients 10 --num_data_per_user 512 --local_bs 64 --num_rounds 50 --privacy_budget 4

python3 fed_dp.py --num_users 5 --num_chosen_clients 5 --num_data_per_user 512 --local_bs 64 --num_rounds 25 --privacy_budget 10 --clipping_threshold 2
python3 fed_dp.py --num_users 5 --num_chosen_clients 5 --num_data_per_user 512 --local_bs 64 --num_rounds 25 --privacy_budget 10 --num_uplink_exposures 5