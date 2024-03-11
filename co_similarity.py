import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path
import pandas as pd
from compute_same import similarity

def get_args_parser():
    parser = argparse.ArgumentParser('Completeness part', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL',
                        help='Name of model to use')
    # Dataset parameters
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--input_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--text_descriptions', default='amazon', type=str, 
                        help='name of the evalauted text set')
    parser.add_argument('--dataset', type=str, default='waterbirds', 
                        help='imagenet or waterbirds')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--num_of_last_layers', type=int, default=8, 
                        help="How many attention layers to replace.")
    parser.add_argument('--texts_per_head', type=int, default=31, help='The number of text examples per head.')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--lr')
    parser.add_argument('--layer',type=int)
    return parser

def main(args):
    with open(os.path.join(args.input_dir, f'{args.dataset}_{args.texts_per_head}span_{args.layers}_{args.model}_{args.lr}_Orthogonal.npy'), 'rb') as f:
            #    span.append(np.load(f))
            #with open(os.path.join(args.input_dir, f'{dataset_list[i]}_{head_num_list[t]}span_{args.layers}_{args.model}_{args.lr}.npy'), 'rb') as f:
        span = np.load(f)
        import pdb;pdb.set_trace()
    layers_dict = []
    for i in range(args.layers):
        dict = {}
        for n in range(15):
            for m in range(n+1, 16):
                dict[(n, m)] = similarity(span[i][n], span[i][m], span.shape[2])
        layers_dict.append(dict)
    with open(os.path.join(args.input_dir, f'co_similarity_{args.dataset}_{args.texts_per_head}_head_in_all_layers_{args.model}.txt'),'w') as w:
        for i in range(len(layers_dict)):
            w.write(f'layer {i}\n')
            for n in range(15):
                for m in range(n+1,16):
                    w.write(f'similarity between head {n} and head {m} : {dict[(n,m)]}\n')


    sorted_keys = sorted(layers_dict[args.layer], key=layers_dict[args.layer].get)
    sum = 0
    for key in sorted_keys:
        sum = sum + layers_dict[args.layer][key]
        print(f'{key} similarity: {layers_dict[args.layer][key]}')
    print(sum/120)
        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)