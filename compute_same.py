import time
import numpy as np
import torch
import glob
import sys
import os
import einops
from torch.utils.data import DataLoader
import tqdm
import argparse
from pathlib import Path
import pandas as pd

def similarity(t1, t2, n):
    p = (t1.dot(t2.T))/(np.linalg.norm(t1, axis=1).reshape(-1, 1)*np.linalg.norm(t2, axis=1))
    dp = np.zeros((n, n))
    dp[0] = p[0]
    # 递推计算最大总和
    for i in range(1, n):
        for j in range(n):
            dp[i, j] = max(dp[i-1, k] for k in range(n) if k != j) + p[i, j]
    return max(dp[-1])

def main():
    t1 = torch.randn(20, 1024)
    t2 = torch.randn(20, 1024)
    a = similarity(t1, t2, 20)
    b = similarity(t2, t1, 20)
    print(a, b)

if __name__ == '__main__':
    main()

    