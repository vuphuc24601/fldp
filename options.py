#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument(
        "--num_rounds", type=int, default=25, help="number of fl rounds"
    )
    parser.add_argument("--start_round", type=int, default=0, help="start round of fl")

    parser.add_argument("--num_users", type=int, default=50, help="number of users: N")
    parser.add_argument(
        "--num_chosen_clients", type=int, default=50, help="number of users: K"
    )
    parser.add_argument(
        "--local_ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=512, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--num_data_per_user", type=int, default=512)

    # model arguments
    parser.add_argument("--model", type=str, default="mlp", help="model name")

    # dp arguments
    parser.add_argument("--privacy_budget", type=int, default=50)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument(
        "--clipping_threshold", type=int, default=1, help="clipping threshold"
    )
    parser.add_argument(
        "--num_uplink_exposures",
        type=int,
        default=4,
        help="number of uplink channels exposures: L",
    )

    # other arguments
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument("--dataset_used", type=str, default="/mnist.csv")

    parser.add_argument(
        "--iid",
        type=int,
        default=1,
        help="whether i.i.d or not, 1 for iid, 0 for non-iid",
    )
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="verbose print, 1 for True, 0 for False"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--algorithm", type=str, default="fedavg", help="algorithm for optimization"
    )

    args = parser.parse_args()

    args.loss_fn = torch.nn.CrossEntropyLoss()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_file_baseline = (
        f"results/baseline/N{args.num_users}_"
        f"K{args.num_chosen_clients}_"
        f"DPC{args.num_data_per_user}_"
        f"BS{args.local_bs}_"
        f"T{args.num_rounds}_"
        f"C{args.clipping_threshold}.pkl"
    )

    args.out_file_dp = (
        f"results/dp/N{args.num_users}_"
        f"K{args.num_chosen_clients}_"
        f"DPC{args.num_data_per_user}_"
        f"BS{args.local_bs}_"
        f"E{args.privacy_budget}_"
        f"T{args.num_rounds}_"
        f"C{args.clipping_threshold}.pkl"
    )

    return args
