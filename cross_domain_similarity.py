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
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--input_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--w_ov_rank', type=int, default=80, help='The rank of the OV matrix')
    parser.add_argument('--texts_per_head', type=int, default=10, help='The number of text examples per head.') #
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    parser.add_argument('--dataset_list') #
    parser.add_argument('--n', type=int) #
    parser.add_argument('--layers', type=int) #
    parser.add_argument('--lr')
    parser.add_argument('--head_num',type=int)
    parser.add_argument('--head_num_list')
    return parser

def main(args):
    all_select = [] #[t, 3, 24, 16]
    dataset_list = args.dataset_list.split(',')
    head_num_list = args.head_num_list.split(',')
    head_num_list = [int(num) for num in head_num_list]
    with open(os.path.join(args.input_dir, f'cross_{args.n}_domain_similarity_{args.head_num}_head_in_last_{args.layers}_layers_{args.model}.txt'), 'w') as w:
        for t in range(args.head_num): 
            span = []
            for i in range(args.n):
                with open(os.path.join(args.input_dir, f'{dataset_list[i]}_{head_num_list[t]}span_{args.layers}_{args.model}_{args.lr}_Orthogonal.npy'), 'rb') as f:
                #    span.append(np.load(f))
                #with open(os.path.join(args.input_dir, f'{dataset_list[i]}_{head_num_list[t]}span_{args.layers}_{args.model}_{args.lr}.npy'), 'rb') as f:
                    span.append(np.load(f)) # [l, h, texts_span, d]
            sum_list = []
            for i in range(args.n-1):
                for j in range(i+1, args.n):
                    sum_layer_list = []
                    w.write(f'for last {args.layers} layers between {dataset_list[i]} and {dataset_list[j]} in {head_num_list[t]} head\n')
                    print(f'for last {args.layers} layers between {dataset_list[i]} and {dataset_list[j]} in {head_num_list[t]} head\n')
                    for m in range(24-args.layers, 24):
                        sum = 0
                        k = 0
                        for n in range(span[i].shape[1]):
                            sum = sum + similarity(span[i][m, n], span[j][m, n], head_num_list[t])
                            k = k + 1
                            w.write(f'{similarity(span[i][m, n], span[j][m, n], head_num_list[t])}\n')
                            print(f'{similarity(span[i][m, n], span[j][m, n], head_num_list[t])}')
                        print('\n')
                        w.write('\n')
                        sum_layer_list.append(sum/k)
                    sum_list.append(sum_layer_list)
            
            ##for i in range(len(sum_list)):
            num_head = []
            k = 0
            for i in range(args.n-1):
                for j in range(i+1, args.n):
                    list_small = []
                    o = 0
                    layers_head = []
                    for m in range(24-args.layers,24):
                        add = 0
                        layer_head = []
                        for n in range(span[i].shape[1]):
                            #print(f'{similarity(span[i][m, n], span[j][m, n], args.texts_per_head) > sum_list[k][o]}', end=' ')
                            if(similarity(span[i][m, n], span[j][m, n], head_num_list[t]) > sum_list[k][o]):
                                add = add + 1
                                layer_head.append(1)
                            else: 
                                layer_head.append(0)
                        #print('\n')
                        #print(sum_list[k][o],'\n')
                        o = o + 1
                        list_small.append(add)
                        layers_head.append(layer_head)
                    #print(list_small)
                    k = k + 1
                    num_head.append(layers_head)
            all_select.append(num_head)


        for k in range(int((args.n*(args.n-1))//2)):
            print(f'domain_conbination_{k}')
            w.write(f'domain_conbination_{k}\n')
            for i in range(24-args.layers, 24):
                print(f'layer{i}',end='\n')
                w.write(f'layer{i}\n')
                for j in range(args.head_num):
                    for n in range(16):
                        w.write(f'{all_select[j][k][i][n]} ')
                        print(f'{all_select[j][k][i][n]}',end=' ')
                    print('\n')
                    w.write('\n')
                  
        


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
        