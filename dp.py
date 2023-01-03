import numpy as np
import torch


def get_noise_scale(args):
    L = args.num_rounds
    c = np.sqrt(2*np.log(1.25/args.delta))
    Delta_s = 2*args.clipping_threshold/args.num_data_per_user

    noise_scale = c*Delta_s*L/args.privacy_budget

    return noise_scale

def add_noise(args, noise_scale, w):
    for k in w.keys():
        noise = np.random.normal(0, noise_scale, w[k].size())
        noise = torch.from_numpy(noise)
        w[k] += noise

    return w

def get_server_noise_scale(args):
    c = np.sqrt(2 * np.log(1.25 / args.delta))
    noise_scale = 2 * c * args.clipping_threshold * np.sqrt((args.num_rounds**2)-(args.num_uplink_exposures**2)*args.num_users)/(args.num_data_per_user*args.num_users*args.privacy_budget)
    return noise_scale

