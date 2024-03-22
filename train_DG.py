import numpy as np
import torch
from PIL import Image
import os.path
import loralib
import argparse
import random
from pathlib import Path
from torchviz import make_dot
import torch.nn as nn
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from prs_hook import hook_prs_logger
from torchvision.datasets import ImageNet, ImageFolder
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.binary_waterbirds import BinaryWaterbirds
from torchvision.datasets import CIFAR100, CIFAR10
from sklearn.model_selection import train_test_split
from utils.generatelist import generate_classlist_and_labels



parser = argparse.ArgumentParser('Project Residual Stream', add_help=False)
parser.add_argument('--bs', default=2, type=int,
                    help='Batch size')
# Model parameters
parser.add_argument('--model', default='ViT-L-14', type=str, metavar='MODEL',
                    help='Name of model to use')
parser.add_argument('--pretrained', default='laion2b_s32b_b82k', type=str)
# Dataset parameters
parser.add_argument('--dataset_root_path', default = '/cephfs/tianlei/dataset/')
parser.add_argument('--dataset')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--output_dir', default='./output_dir',
                    help='path where to save')
parser.add_argument('--input_dir', default='./output_dir')
parser.add_argument('--device',default='cuda:0',
                    help='device to use for testing')
parser.add_argument('--domain')
parser.add_argument('--get_rate',type=float)
parser.add_argument('--dataset_name')
parser.add_argument('--epochs',type=int)
parser.add_argument('--seed',type=int)
parser.add_argument('--model_path',default='/cephfs/tianlei/models/')
parser.add_argument('--lr',type=float)
args = parser.parse_args()

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

def generation_loader(m, datasets):
    target_loader = DataLoader(datasets[m], batch_size=args.bs, shuffle=True, num_workers=8)
    k = datasets.copy()
    del k[m]
    #import pdb;pdb.set_trace()
    source_sets = torch.utils.data.ConcatDataset(k)
    source_loader = DataLoader(source_sets, batch_size=args.bs, shuffle=True, num_workers=8)
    return source_loader, target_loader
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test(target_loader, model, device, classifier):
    total_correct = 0
    total = 0
    with torch.no_grad():
        for i,(x, y) in tqdm(enumerate(target_loader), total = len(target_loader)):
            x = x.to(device)
            y = y.to(device)
            img_features = model.encode_image(x.to(args.device), attn_method='head', normalize=False)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            similarity =  (100.0 * img_features @ classifier).softmax(dim=-1)
            pred = torch.argmax(similarity,dim=-1,keepdim=False)
            correct = torch.sum(pred==y)
            total_correct += correct
            total += x.shape[0]
        test_acc = total_correct/total
    return test_acc     
     

model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
domain = args.domain.split(',')
with open(os.path.join(args.input_dir, f'{args.dataset_name}_classifier_{args.model}.npy'), 'rb') as f:
    classifier = np.load(f)
    classifier = torch.tensor(classifier).to(args.device)
datasets = []
for i in range(len(domain)):
    datasets.append(ImageFolder(root=os.path.join(args.dataset_root_path, args.dataset, domain[i]), transform=preprocess))
criterion = torch.nn.CrossEntropyLoss() 

set_seed(args.seed)
#with open(os.path.join(args.output_dir, f'{args.dataset}_da_{args.get_rate}.txt'), 'w') as w:
with open(os.path.join(args.output_dir, f'{args.dataset}_dg_{args.get_rate}_one_head_adamw_{args.lr}_{args.model}.txt'), 'w') as w: #
    for m in tqdm(range(len(domain)),total=len(domain)):
                model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, device=args.device)
                model.to(args.device)
                model.init_lora() #先set_lora后set_gate在源域上进行训练，接着得到目标域上的gate特征
                model.to(args.device)
                loralib.utils.mark_only_lora_as_trainable(model=model)
                optim = torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-4)
                optimw = torch.optim.AdamW(model.parameters(),lr=args.lr,eps=1e-6)
                optimsgd = torch.optim.SGD(model.parameters(), lr=args.lr)
                source_loader, target_loader = generation_loader(m, datasets)
                acc = []
                for epoch in range(args.epochs):  
                    best = [0,0]
                    model.train()
                    total_loss = 0
                    total = 0
                    for i,(x,y) in tqdm(enumerate(source_loader),total=len(source_loader)):
                        x,y = x.to(args.device),y.to(args.device)
                        img_features = model.encode_image(x,normalize=False,attn_method='head')
                        norm = img_features.norm(dim=-1,keepdim=True)  
                        similarity =  (100.0 * (img_features/norm) @ classifier)
                        loss = criterion(similarity,y)
                        optimw.zero_grad()
                        loss.backward()
                        optimw.step()
                        total += 1
                        total_loss += loss.item()
                    model.eval()
                    current_acc = test(target_loader,model,args.device,classifier)
                    if current_acc > best[1]:
                        best[0] = epoch
                        best[1] = current_acc
                        torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'to {domain[m]}_one_head_adamw{args.model}''.pt')#
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    w.write(f'to {domain[m]} :epoch: {epoch} loss: {total_loss/total:.4f} current_acc: {current_acc:.4f} current_best: {best[1]} best_epoch: {best[0]}\n')
                    acc.append(current_acc.cpu())
                plt.plot( acc, marker='o', linestyle='-',label=f'to {domain[m]}')
    plt.xlabel('epoch')   
    plt.ylabel('acc')
    plt.legend()
    #plt.savefig(f'DG in {args.dataset}_mixdata_{args.get_rate}_one_head_sgd_{args.lr}.png')#
    plt.savefig(f'DG in {args.dataset}_mixdata_{args.get_rate}_{args.model}_adamw.png')#