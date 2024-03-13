import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import random
from tqdm import tqdm
from utils.misc import accuracy
import pandas as pd
from itertools import combinations
from utils.generatelist import generate_classlist_and_labels
import copy
import matplotlib.pyplot as plt
from compute_same import similarity

def get_args_parser():
    parser = argparse.ArgumentParser('Ablations part', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL',
                        help='Name of model to use')
    # Dataset parameters
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--figures_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--input_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--dataset', type=str, default='waterbirds_binary', 
                        help='imagenet, waterbirds, waterbirds_binary or cub')
    parser.add_argument('--dataset_name')
    parser.add_argument('--lr',default=0,type = int)
    parser.add_argument('--texts_per_head', default=20,type=int)
    return parser
    
    
def main(args):
    with open(os.path.join(args.input_dir, f'{args.dataset}_attn_{args.model}.npy'), 'rb') as f:
        attns = np.load(f) # [b, l, h, d]
    with open(os.path.join(args.input_dir, f'{args.dataset}_mlp_{args.model}.npy'), 'rb') as f:
        mlps = np.load(f) # [b, l+1, d]
    #with open(os.path.join(args.input_dir, f'{args.dataset}_{args.texts_per_head}span_{args.layers}_{args.model}_{args.lr}_Orthogonal.npy'), 'rb') as f:
        #representation = np.load(f) # [l, h, text, d]
    attns = np.sum(attns, axis=2,keepdims=True)
    for i in range(attns.shape[3]):
        for j in range(attns.shape[1]):
            attns[:, j, :, i] = np.mean(attns[:, j, :, i],axis=None,keepdims=False)   
    for i in range(mlps.shape[1]):
        for j in range(mlps.shape[2]):
            mlps[:, i, j] = np.mean(mlps[:, i, j],axis=None,keepdims=False)
    attns_layer = []
    mlps_layer = []
    layer = []
    sum = mlps[0, 0, :]
    for i in range(attns.shape[1]):
        sum += attns[0, i, 0, :]
        attns_layer.append(1 - similarity(np.expand_dims(sum - attns[0, i, 0, :],axis=0),np.expand_dims(sum,axis=0), 1))
        sum += mlps[0, i + 1, :]
        mlps_layer.append(1 - similarity(np.expand_dims( sum - mlps[0, i + 1, :],axis=0),np.expand_dims(sum,axis=0), 1))
        layer.append(1 - similarity(np.expand_dims( sum - mlps[0, i + 1, :] - attns[0, i, 0, :],axis=0),np.expand_dims(sum,axis=0), 1))
    plt.plot(mlps_layer, marker='o', linestyle='-',label='mlp_layer_Contribution')
    plt.plot(attns_layer, marker='o', linestyle='-',label='attns_layer_Contribution')
    plt.xlabel('layers')
    plt.ylabel('Contribution')
    plt.legend()
    plt.savefig(f'layers_Contribution_in_{args.dataset}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)
