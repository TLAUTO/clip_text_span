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

def mark_only_soft_mask_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'mask' not in n:
            p.requires_grad = False

def adaptaion_loader(get_rate, source_target, dataset):
    total_target_length = len(dataset[source_target[1]])
    get_length = int(get_rate*total_target_length)
    target_length = total_target_length - get_length
    st, t = torch.utils.data.random_split(dataset[source_target[1]], [get_length, target_length])
    #st_dataset = torch.utils.data.ConcatDataset([st, dataset[source_target[0]]])
    s_loader = DataLoader(dataset[source_target[0]], batch_size=args.bs, shuffle=True, num_workers=8)
    st_dataloader = DataLoader(st, batch_size=args.bs, shuffle=True, num_workers=8)
    t = DataLoader(t, batch_size=args.bs, shuffle=True, num_workers=8)
    return s_loader, st_dataloader, t

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
            img_features = model.encode_image(x, attn_method='head', normalize=False)
            img_features /= img_features.norm(dim=-1,keepdim=True)
            similarity =  (100.0 * img_features @ classifier).softmax(dim=-1)
            pred = torch.argmax(similarity,dim=-1,keepdim=False)
            correct = torch.sum(pred==y)
            total_correct += correct
            total += x.shape[0]
        test_acc = total_correct/total
    return test_acc     

def mark_only_gate_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'gate' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    

domain = args.domain.split(',')
url = args.dataset_name+'_'+f'{domain[0]}_to_{domain[1]}_adamw_128''.pt'

model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, url_dict={'url': f'{url}', 'hf_hub': ''}, cache_dir='/cephfs/tianlei/models', url=True, is_lora=True, device=args.device)

with open(os.path.join(args.input_dir, f'{args.dataset_name}_classifier_{args.model}.npy'), 'rb') as f:
    classifier = np.load(f)
    classifier = torch.tensor(classifier).to(args.device)
dataset = []
for i in range(len(domain)):
    dataset.append(ImageFolder(root=os.path.join(args.dataset_root_path, args.dataset, domain[i]), transform=preprocess))
    
optim = torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-4)
criterion = torch.nn.CrossEntropyLoss() 
#start
set_seed(args.seed)
#with open(os.path.join(args.output_dir, f'{args.dataset}_da_{args.get_rate}_uperbandgate_one_head.txt'), 'w') as w:
with open(os.path.join(args.output_dir, f'{args.dataset}_da_{args.get_rate}_uperbandgate_mask_one_head.txt'), 'w') as w: #change

#with open(os.path.join(args.output_dir, f'{args.dataset}_da_{args.get_rate}_uperbandgate_mask.txt'), 'w') as w:
#with open(os.path.join(args.output_dir, f'{args.dataset}_da_{args.get_rate}_uperbandgate.txt'), 'w') as w:
    for m in tqdm(range(len(domain)-1),total=len(domain)-1):
        for n in range(m+1, len(domain)):
            for k in range(2):
                best = [0,0]
                if k:
                    source_loader, mini_loader, target_loader = adaptaion_loader(args.get_rate, [n,m], dataset)
                    url = args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}_adamw_128''.pt'# (one_head)
                    #url = args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}''.pt'#
                else:
                    source_loader, mini_loader, target_loader = adaptaion_loader(args.get_rate, [m,n], dataset)
                    url = args.dataset_name+'_'+f'{domain[m]}_to_{domain[n]}_adamw_128''.pt'# (one_head)
                    #url = args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}''.pt'#
                acc = []
                for epoch in range(args.epochs):  
                    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained, url_dict={'url': f'{url}', 'hf_hub': ''}, cache_dir='/cephfs/tianlei/models', url=True, device=args.device, is_lora=True)
                    model.to(args.device)
                    #model.init_lora()
                    #model.init_gate()
                    model.init_soft_gate() #先set_lora后set_gate在源域上进行训练，接着得到目标域上的gate特征
                    model.to(args.device)
                    #mark_only_gate_as_trainable(model=model)
                    mark_only_soft_mask_as_trainable(model=model)
                    model.train()
                    for name, param in model.named_parameters():
                        if 'mask' in name:
                            print(f"Parameter name: {name}")
                            print(f"Parameter value:\n{param}")
                    total_loss = 0
                    total = 0
                    for i,(x,y) in tqdm(enumerate(mini_loader),total=len(mini_loader)):
                        x,y = x.to(args.device),y.to(args.device)
                        img_features = model.encode_image(x,normalize=False,attn_method='head')
                        norm = img_features.norm(dim=-1,keepdim=True)  
                        similarity =  (100.0 * (img_features/norm) @ classifier)
                        loss = criterion(similarity,y)
                        optimw = torch.optim.AdamW(model.parameters(),lr=args.lr,eps=1e-6)
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
                        if k:
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}_{args.get_rate}_uperbandgate_one_head''.pt')# change
                            torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}_{args.get_rate}_uperbandgate_mask_one_head''.pt')# change
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}_{args.get_rate}_uperbandgate''.pt')# change
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[n]}_to_{domain[m]}_{args.get_rate}_uperbandgate_mask''.pt')# change
                        else:
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[m]}_to_{domain[n]}_{args.get_rate}_uperbandgate_one_head''.pt')# change
                            torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[m]}_to_{domain[n]}_{args.get_rate}_uperbandgate_mask_one_head''.pt')# change
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[m]}_to_{domain[n]}_{args.get_rate}_uperbandgate''.pt')# change
                            #torch.save(model.state_dict(),args.model_path+args.dataset_name+'_'+f'{domain[m]}_to_{domain[n]}_{args.get_rate}_uperbandgate_mask''.pt')# change
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    if k:
                        w.write(f'{domain[n]} to {domain[m]}_{args.get_rate}_gate:epoch: {epoch} loss: {total_loss/total:.4f} current_acc: {current_acc:.4f} current_best: {best[1]} best_epoch: {best[0]}\n')
                        w.write(f'{model.get_gate()}\n')
                    else:
                        w.write(f'{domain[m]} to {domain[n]}_{args.get_rate}_gate:epoch: {epoch} loss: {total_loss/total:.4f} current_acc: {current_acc:.4f} current_best: {best[1]} best_epoch: {best[0]}\n')
                        w.write(f'{model.get_gate()}\n')
                    w.write('\n')
                    acc.append(current_acc.cpu())
                if k:
                    plt.plot( acc, marker='o', linestyle='-',label=f'{args.get_rate}_{domain[m]}_gate_tuning')
                else:
                    plt.plot( acc, marker='o', linestyle='-',label=f'{args.get_rate}_{domain[n]}_gate_tuning')
    plt.xlabel('epoch')   
    plt.ylabel('acc')
    plt.legend()
    #plt.savefig(f'gate_fit_uperband in {args.dataset}_mixdata_{args.get_rate}_one_head.png')# change
    plt.savefig(f'gate_fit_uperband_mask in {args.dataset}_mixdata_{args.get_rate}_one_head.png')# change
    #plt.savefig(f'gate_fit_uperband in {args.dataset}_mixdata_{args.get_rate}.png')# change
    #plt.savefig(f'gate_fit_uperband_mask in {args.dataset}_mixdata_{args.get_rate}.png')# change