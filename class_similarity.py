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
    parser.add_argument('--n', type=int)
    parser.add_argument('--dataset_list')
    return parser

def main(args):
    dataset_list = args.dataset_list.split(',')
    # Load text:    
    with open(os.path.join(args.input_dir, f'{args.text_descriptions}_{args.model}.npy'), 'rb') as f:
        text_features = np.load(f)
    span = []
    aver_layers = []
    with open(os.path.join(args.input_dir, f'class_similarity_in_office31_domain_last_{args.layers}_layers_{args.model}.txt'), 'w') as w:
        for i in range(args.n):
            with open(os.path.join(args.input_dir, f'{dataset_list[i]}_{args.texts_per_head}span_{args.layers}_{args.model}_{args.lr}_Orthogonal.npy'), 'rb') as f:
            #    span.append(np.load(f))
            #with open(os.path.join(args.input_dir, f'{dataset_list[i]}_{head_num_list[t]}span_{args.layers}_{args.model}_{args.lr}.npy'), 'rb') as f:
                span.append(np.load(f))
            sum_layer_list = []    
            w.write(f'{dataset_list[i]}\n')
            for m in range(24-args.layers,24):
                sum = 0
                k = 0
                for n in range(span[i].shape[1]):
                    w.write(f'{similarity(span[i][m, n], text_features, args.texts_per_head)}\n')
                    print(similarity(span[i][m, n], text_features, args.texts_per_head))
                    sum = sum + similarity(span[i][m, n], text_features, args.texts_per_head)
                    k = k + 1
                w.write('\n')
                print('\n')
                sum_layer_list.append(sum/k)
            aver_layers.append(sum_layer_list)
        
        for i in range(args.n):
            w.write(f'{dataset_list[i]}\n')
            print(f'{dataset_list[i]}')
            for m in range(24-args.layers,24):
                for n in range(16):
                    w.write(f'{similarity(span[i][m, n], text_features, args.texts_per_head) > aver_layers[i][m]} ')
                    print(similarity(span[i][m, n], text_features, args.texts_per_head) > aver_layers[i][m],end=' ')
                print('\n')
                w.write('\n')
    



        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)