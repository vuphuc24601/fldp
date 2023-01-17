import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import server
from client import client_train
from models import MLP, Net
from options import args_parser
from utils import get_mnist_iid, load_data

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


def fldp(args):
    if args.dataset == "mnist":
        dataset_train, dataset_test = load_data()
    else:
        # TODO
        pass

    if args.iid:
        dict_users = get_mnist_iid(
            dataset_train, args.num_users, args.num_data_per_user
        )
    else:
        # TODO
        pass

    train_loaders = []
    for client in range(args.num_users):
        tem_dataset = Subset(dataset_train, dict_users[client])
        train_loaders.append(
            DataLoader(tem_dataset, batch_size=args.local_bs, shuffle=True)
        )

    val_loader = DataLoader(dataset_test, batch_size=args.local_bs, shuffle=False)

    # Load model
    if args.model == "mlp":
        global_param = MLP().state_dict()
    elif args.model == "net":
        global_param = Net().state_dict()
    else:
        # TODO
        pass

    # save loss and acc in result
    res = {}
    for k1 in ("train", "val"):
        res[k1] = {}
        for k2 in ("loss", "acc", "precision", "recall", "f1"):
            res[k1][k2] = {}
            res[k1][k2]["avg"] = []
            for k3 in range(args.num_users):
                res[k1][k2][k3] = []

    # start federated learning
    for round in range(args.start_round, args.num_rounds):
        start = time.time()
        print(f"Round {round+1}/{args.num_rounds}")

        # metrics
        train_loss, test_loss = 0, 0
        train_corr, test_acc = 0, 0
        train_total = 0
        list_params = []

        chosen_clients = random.sample(range(args.num_users), args.num_chosen_clients)
        chosen_clients.sort()

        for i in tqdm(chosen_clients):
            print(f"-----------client {i} starts training----------")
            tem_param, train_summ = client_train(
                args, deepcopy(global_param), train_loaders[i], epochs=args.local_ep
            )

            train_loss += train_summ["loss"]
            train_corr += train_summ["correct"]
            train_total += train_summ["total"]

            list_params.append(tem_param)

        res["train"]["loss"]["avg"].append(train_loss / args.num_users)
        res["train"]["acc"]["avg"].append(train_corr / train_total)

        print(
            "Train loss {:5f} acc {:5f}".format(
                res["train"]["loss"]["avg"][-1], res["train"]["acc"]["avg"][-1]
            )
        )

        # Server aggregation
        global_param = server.FedAvg(list_params)
        val_summ = server.test(args, global_param, val_loader)

        res["val"]["loss"]["avg"].append(val_summ["loss"])
        res["val"]["acc"]["avg"].append(val_summ["correct"] / val_summ["total"])

        print(
            "Val   loss: {:5f} acc: {:5f}".format(
                res["val"]["loss"]["avg"][-1],
                res["val"]["acc"]["avg"][-1],
            )
        )

        for i in range(args.num_users):
            summ = server.test(args, global_param, train_loaders[i])

            res["val"]["loss"][i].append(summ["loss"])
            res["val"]["acc"][i].append(summ["correct"] / summ["total"])

        # folder_path = f"./results/models/fl/" \
        #                 f"N{args.num_users}_" \
        #                 f"K{args.num_chosen_clients}_" \
        #                 f"DPC{args.num_data_per_user}_" \
        #                 f"BS{args.local_bs}_" \
        #                 f"T{args.num_rounds}_" \
        #                 f"C{args.clipping_threshold}"
        #
        # os.makedirs(folder_path, exist_ok=True)
        # torch.save(
        #     global_param,
        #     f"{folder_path}/round{round}.pt"
        # )
        print("Time {}".format(time.time() - start))
        print()

    with open(args.out_file_baseline, "wb") as fp:
        pickle.dump(res, fp)


if __name__ == "__main__":
    args = args_parser()
    fldp(args)
