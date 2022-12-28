from copy import deepcopy

import torch

from metrics import Metric
from models import MLP


def FedAvg(list_params):
    agg_param = deepcopy(list_params[0])
    for k in agg_param.keys():
        agg_param[k] = torch.stack([param[k].float() for param in list_params], 0).mean(
            0
        )
    return agg_param


def test(args, param, loader):
    """
    Evaluate the scheme
        - args: configuration
        - param: model state dict
        - loader: test set
    """

    if args.dataset == "mnist":
        # print("model used: mnist")
        model = MLP().to(args.device)

    model.load_state_dict(param)
    model.eval()

    summ = Metric()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)
            loss = args.loss_fn(output, target)
            summ.update(output.argmax(dim=1).detach().cpu(), target.cpu(), loss.item())

    return summ.get()
