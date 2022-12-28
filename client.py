from copy import deepcopy
from random import randint

import torch

from metrics import Metric
from models import MLP


def client_train(args, param, loader, epochs):
    """
    Training on each client
        - args: configuration
        - param: global param
        - loader: client dataset for training
        - epochs: number of training iterations
    """
    # load model
    if args.model == "mlp":
        model = MLP().to(args.device)

    model.load_state_dict(param)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.train()

    metric = Metric()

    for epoch in range(epochs):
        for data, target in loader:
            if len(data) == 1:  # Avoid error when using BatchNorm
                continue
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = args.loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Only keep track of the last epoch
            if epoch == epochs - 1:
                metric.update(
                    output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item()
                )

    return deepcopy(model.cpu().state_dict()), metric.get()
