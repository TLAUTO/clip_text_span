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

def get_args_parser():
    parser = argparse.ArgumentParser('Ablations part', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL',
                        help='Name of model to use')
    # Dataset parameters
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset_dir')
    parser.add_argument('--figures_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--input_dir', default='./output_dir',
                        help='path where data is saved')
    parser.add_argument('--dataset', type=str, default='waterbirds_binary', 
                        help='imagenet, waterbirds, waterbirds_binary or cub')
    parser.add_argument('--random_num')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dataset_name')
    return parser
    
    
def main(args):
    if args.model == 'ViT-H-14':
        to_mean_ablate_setting = [(31, 12), (30, 11), (29, 4)]
        to_mean_ablate_geo = [(31, 8), (30,15), (30, 12), (30, 6), (29, 14), (29, 8)]
    elif args.model == 'ViT-L-14':
        random_num = int(args.random_num)
        random_ablation = [(random.randint(23 - args.layers + 1, 23), random.randint(0, 15)) for _ in range(random_num)]
        to_enhence_output = [

        ]
        to_mean_ablate_geo = [ (23,15)]
                              #(21,15),(21,2),
                              #(20, 3),(20, 8),(20,15),(20,14),
                              #(19,1),(19,14),(19,13),(19,8),(19,12)
        to_mean_ablate_setting = []
                            # (19, 2),(19, 3),(19,8),(19,9),(19,10),(19,11),(19,14),(19,15),
                            #    (18,1),(18,5),(18,10),(18,13),(18,15),
                            #    (17,0),(17,9),(17,4),
                            #      (16,1),(16,2),(16,3),(16,4),(16,6),(16,8),(16,10),(16,11),(16,14),(15,0),(15,1),(15,2),
                             ###     (15,3),(15,6),(15,7),(15,9),(15,10),
                             #     (14, 12),
                              #    (13, 0),(13,1),(13,2),(13,4),(13,5),(13,6),(13,7),(13,9),(13,10),(13,11),(13,12),
                             #     (12,0),(12,3),(12,7),(12,8),(12,11),(12,12),(12,13),(12,14),(12,15),
                             #     (11,0),(11,2),(11,4),(11,5),(11,6),(11,7),(11,8),(11,9),(11,12),(11,13),
                             #     (10,0),(10,1),(10,3),(10,4),(10,5),(10,6),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(10,14),(10,15),
                             #     (9,3),(9,4),(9,13),(9,14),

    else: 
        assert args.model == 'ViT-B-16'
        to_mean_ablate_setting = [(11, 3), (10, 11), (10, 10), (9, 8), (9, 6)]
        to_mean_ablate_geo = [(11, 6), (11, 0)]
    to_mean_ablate_output =  to_mean_ablate_geo + to_mean_ablate_setting
    with open(os.path.join(args.input_dir, f'{args.dataset}_attn_{args.model}.npy'), 'rb') as f:
        attns = np.load(f) # [b, l, h, d]
    with open(os.path.join(args.input_dir, f'{args.dataset}_mlp_{args.model}.npy'), 'rb') as f:
        mlps = np.load(f) # [b, l+1, d]
    with open(os.path.join(args.input_dir, f'{args.dataset_name}_classifier_{args.model}.npy'), 'rb') as f:
        classifier = np.load(f)
    
    if args.dataset == 'imagenet':
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    else:
        ## with open(os.path.join('waterbird_complete95_forest2water2', 'metadata.csv'), 'rb') as f:
        '''with open(os.path.join('waterbird_complete95_forest2water2', 'metadata.csv'), 'rb') as f:
            df = pd.read_csv(f)
            import pdb;pdb.set_trace()
            labels = df.iloc[:, 2].values
            print(len(labels))
            labels = np.array(labels)'''        
        '''with open(os.path.join('waterbird_complete95_forest2water2', 'metadata.csv'), 'rb') as f:
            # labels = np.load(f)
            df = pd.read_csv(f)
            labels = df.iloc[:, 1].values
            # 使用列表推导将每个元素的前三个字符转换为整数
            int_labels_array = np.array([int(label[:3]) for label in labels])
            # 输出转换后的数组
            labels = np.array(int_labels_array)-1'''
        _, labels, _ = generate_classlist_and_labels(args.dataset_dir)
        labels = np.array(labels)
    print(f'ablation num :{random_num}')
    for layer in range(23 - args.layers + 1):  #平均前面的层
        for head in range(attns.shape[2]):
            attns[:, layer, head, :] = np.mean(attns[:, layer, head, :], axis=0, keepdims=True)
    #for layer in range(mlps.shape[1]): # 消融mlp层
    #    mlps[:, layer] = np.mean(mlps[:, layer], axis=0, keepdims=True)
    baseline = attns.sum(axis=(1,2)) + mlps.sum(axis=1)
    baseline_acc = accuracy(torch.from_numpy(baseline @ classifier).float(), 
                            torch.from_numpy(labels))[0]*100
    print('Baseline:', baseline_acc)
    y1 = []
    y2 = []
    base = []
    a = copy.deepcopy(mlps)
    b = copy.deepcopy(mlps)
    for i in range( mlps.shape[1]):
        a[:, i] = np.mean(a[:, i], axis=0, keepdims=True)
        mlp_ablation = attns.sum(axis=(1,2)) + a.sum(axis=1)
        mlp_ablation_acc = accuracy(torch.from_numpy(mlp_ablation @ classifier).float(), 
                                torch.from_numpy(labels))[0]*100
        y1.append(mlp_ablation_acc)
        base.append(baseline_acc)
        print(f'mlp_ablation_in_first_{i}_layers:', mlp_ablation_acc)
    plt.plot( y1, marker='o', linestyle='-',label=f'from_first_mlp_ablation_{args.dataset}')
    for i in range(mlps.shape[1]):
        b[:, 24-i] = np.mean(b[:, 24-i], axis=0, keepdims=True)
        mlp_ablation = attns.sum(axis=(1,2)) + b.sum(axis=1)
        mlp_ablation_acc = accuracy(torch.from_numpy(mlp_ablation @ classifier).float(), 
                                torch.from_numpy(labels))[0]*100
        y2.append(mlp_ablation_acc)
    print(f'mlp_ablation_in_first_{i}_layers:', mlp_ablation_acc)
    plt.plot( y2, marker='o', linestyle='-',label=f'from_last_mlp_ablation_{args.dataset}')
    plt.plot( base, marker='o', linestyle='-',label='baseline')
    plt.title(f'mlp_ablation_{args.dataset}')
    plt.xlabel('mlp_ablation_num')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(f'mlp_ablation_{args.dataset}')
    import pdb; pdb.set_trace()
    ran = copy.deepcopy(attns)
    for layer, head in random_ablation:
        ran[:, layer, head, :] = np.mean(ran[:, layer, head, :], axis=0, keepdims=True)
    random_ = ran.sum(axis=(1,2)) + mlps.sum(axis=1)
    random_acc = accuracy(torch.from_numpy(random_ @ classifier).float(), 
                            torch.from_numpy(labels))[0]*100
    print('Random:', random_acc)
    list_all = []
    k = args.layers*16
    a = 23-args.layers+1
    b = 0
    while k > 0 :
        if b == 15:
            list_all.append((a,b))
            b = 0
            a = a + 1
        else:
            list_all.append((a,b))
            b = b + 1
        k = k - 1
    combs = list(combinations(list_all, random_num))
    dict = {}

    to_mean_ablate_output =  to_mean_ablate_geo + to_mean_ablate_setting
    for layer, head in to_mean_ablate_output:
        ran = copy.deepcopy(attns)
        #attns[:, layer, head, :] = np.zeros((attns.shape[0], attns.shape[3]))
        ran[:, layer, head, :] = ran[:, layer, head, :]*0.5#np.mean(attns[:, layer, head, :], axis=0, keepdims=True)
    for layer, head in to_enhence_output:
        ran[:, layer, head, :] = ran[:, layer, head, :]*2
    ablated = ran.sum(axis=(1,2)) + mlps.sum(axis=1)
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), 
                            torch.from_numpy(labels))[0]*100
    print('Replaced:', ablated_acc)
    for i in tqdm(range(len(combs)), desc='进度',position=0,leave=True):
        ran = copy.deepcopy(attns)
        for layer, head in combs[i]:
            ran[:, layer, head, :] = np.mean(ran[:, layer, head, :], axis=0, keepdims=True)
        for layer in range(mlps.shape[1]):
            mlps[:, layer] = np.mean(mlps[:, layer], axis=0, keepdims=True)
        random__ = ran.sum(axis=(1,2)) + mlps.sum(axis=1)
        random_acc = accuracy(torch.from_numpy(random__ @ classifier).float(), 
                                torch.from_numpy(labels))[0]*100
        dict[combs[i]] = random_acc
    sorted_keys = sorted(dict, key=dict.get, reverse=True)
    print(f'upband: {dict[sorted_keys[0]]}')
    y1 = [dict[sorted_keys[i]] for i in range(len(sorted_keys))]
    y2 = [random_acc for _  in range(len(sorted_keys))]
    plt.plot(y1, marker='o', linestyle='-',label='sorted_ablation')
    plt.plot(y2, marker='o', linestyle='-',label='random_ablation')
    plt.xlabel('head')
    plt.ylabel('Acc')
    # 显示图表
    plt.savefig(f'{args.random_num} head ablation in last {args.layers} layers in domain {args.dataset}')
    with open(os.path.join(args.input_dir, f'{args.dataset}_ablation_{args.random_num}_head_in_last_{args.layers}_layers_{args.model}.txt'), 'w') as w:
        w.write('--------------------\n')
        w.write(f'{args.random_num}_head_ablation in last {args.layers} layers\n')
        w.write('--------------------\n')
        for item in sorted_keys:
            w.write(f'ablation {item}: {dict[item]}\n')

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)