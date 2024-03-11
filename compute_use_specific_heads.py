import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import random
import tqdm
from utils.misc import accuracy
import pandas as pd

from utils.generatelist import generate_classlist_and_labels

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
    return parser
    
    
def main(args):
    if args.model == 'ViT-H-14':
        to_mean_ablate_setting = [(31, 12),  (30, 11), (29, 4)]
        to_mean_ablate_geo = [(31, 8), (30,15), (30, 12), (30, 6), (29, 14), (29, 8)]
    elif args.model == 'ViT-L-14':
        to_enhence_output = [

        ]
        to_mean_ablate_geo = [(23,6)
                              
                              
                              
                              
                              ]
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
        
    baseline = attns.sum(axis=(1,2)) + mlps.sum(axis=1)
    baseline_acc = accuracy(torch.from_numpy(baseline @ classifier).float(), 
                            torch.from_numpy(labels))[0]*100
    print('Baseline:', baseline_acc)
    for layer, head in to_mean_ablate_output:
        #attns[:, layer, head, :] = np.zeros((attns.shape[0], attns.shape[3]))
        attns[:, layer, head, :] = np.mean(attns[:, layer, head, :], axis=0, keepdims=True)#np.mean(attns[:, layer, head, :], axis=0, keepdims=True)
    for layer, head in to_enhence_output:
        attns[:, layer, head, :] = attns[:, layer, head, :]*3
    #for layer in range(attns.shape[1]-24):
    #    for head in range(attns.shape[2]):
    #        attns[:, layer, head, :] = np.mean(attns[:, layer, head, :], axis=0, keepdims=True)
    for layer in range(mlps.shape[1]):
        mlps[:, layer] = np.mean(mlps[:, layer], axis=0, keepdims=True)
    ablated = attns.sum(axis=(1,2)) + mlps.sum(axis=1)
    ablated_acc = accuracy(torch.from_numpy(ablated @ classifier).float(), 
                            torch.from_numpy(labels))[0]*100
    print('Replaced:', ablated_acc)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)