#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument("--num_rounds", type=int, default=3, help="number of fl rounds")
    parser.add_argument("--start_round", type=int, default=0, help="start round of fl")

    parser.add_argument("--num_users", type=int, default=4, help="number of users: K")
    parser.add_argument(
        "--num_chosen_clients", type=int, default=4, help="number of users: K"
    )

    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=128, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )

    parser.add_argument("--privacy_budget", type=float, default=50)
    parser.add_argument("--num_data_per_user", type=float, default=1024)

    # model arguments
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument(
        "--kernel_num", type=int, default=9, help="number of each kind of kernel"
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="3,4,5",
        help="comma-separated kernel size to use for convolution",
    )
    parser.add_argument(
        "--norm", type=str, default="batch_norm", help="batch_norm, layer_norm, or None"
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=32,
        help="number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.",
    )
    parser.add_argument(
        "--max_pool",
        type=str,
        default="True",
        help="Whether use max pooling rather than strided convolutions",
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
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--num_channels", type=int, default=3, help="number of channels of imges"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID")
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="verbose print, 1 for True, 0 for False"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--ag_scalar",
        type=float,
        default=1.0,
        help="global aggregation updating scalar, simplicity for A Matrix",
    )
    parser.add_argument(
        "--lg_scalar",
        type=float,
        default=1.0,
        help="client local updating scalar, simplicity for S Matrix",
    )
    parser.add_argument(
        "--algorithm", type=str, default="fedavg", help="algorithm for optimization"
    )

    args = parser.parse_args()

    args.loss_fn = torch.nn.CrossEntropyLoss()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_file = f"results/N{args.num_users}_K{args.num_chosen_clients}_E{args.privacy_budget}_T{args.num_rounds}.pkl"

    return args
