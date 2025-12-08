import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def knn(x, k):
    # x: (B, F, N_nodes)  -->  返回 idx (B, N_nodes, k)
    inner = -2 * torch.matmul(x.transpose(2,1), x)          # (B, N, N)
    xx = torch.sum(x**2, dim=1, keepdim=True)               # (B, 1, N)
    dist = -xx - inner - xx.transpose(2,1)                  # (B, N, N)
    idx = dist.topk(k=k, dim=-1)[1]                         # 取前 k 大 (最小距离)
    return idx                                              # long

def get_graph_feature(x, k=8):
    # x: (B, F, N)  →  return (B, 2F, N, k)
    idx = knn(x, k=k)  # (B, N, k)
    B, F, N = x.size()
    idx_base = torch.arange(0, B, device=x.device).view(-1,1,1)*N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x.transpose(2,1).contiguous().view(B*N, F)  # (B*N, F)
    neighbor = x_flat[idx, :].view(B, N, k, F)           # (B, N, k, F)
    neighbor = neighbor.permute(0,3,1,2)                 # (B, F, N, k)
    x = x.view(B, F, N, 1).repeat(1,1,1,k)
    feature = torch.cat((neighbor - x, x), dim=1)         # (B, 2F, N, k) --> (2F, N, k)
    return feature                                     # 2F channels

# ---------- 3. EdgeConv block ----------
class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=8, bn=True):
        super().__init__()
        self.k = k
        # 输入特征的通道数要是2倍，符合 get_graph_feature 输出 (2*in_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),  # 2*in_channels
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        # 输入：x (B, F, N)，输出：x的特征映射
        feat = get_graph_feature(x, k=self.k)  # 获取图特征
        out = self.mlp(feat).max(dim=-1)[0]  # 采用 max pooling 获取全局特征
        return out

# ---------- 4. DGCNN for binary classification ----------


# 2) 在 DGCNN __init__ 里把 ec1 的 in_channels 改成传进来的值
class DGCNN(nn.Module):
    def __init__(self, in_channels, k=8, emb_dims=256, dropout=0.3):
        super().__init__()
        self.ec1 = EdgeConv(in_channels, 64, k)   # <-- 关键处
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.bn0 = nn.BatchNorm1d(emb_dims)

        self.linear1 = nn.Linear(64+64+128, emb_dims, bias=False)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(emb_dims, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, C, T)  treat T as "nodes"
        x1 = self.ec1(x)                      # (B,64,T)
        x2 = self.ec2(x1)                     # (B,64,T)
        x3 = self.ec3(x2)                     # (B,128,T)

        x_cat = torch.cat((x1.max(-1)[0],     # 每层做全局池化，然后 concat
                           x2.max(-1)[0],
                           x3.max(-1)[0]), dim=1)             # (B,256)

        x = F.leaky_relu(self.bn0(self.linear1(x_cat)))
        x = self.dp1(x)
        x = F.leaky_relu(self.linear2(x))
        return self.linear3(x).squeeze(1)
        # (B,)
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train: optimizer.zero_grad()
        logits = model(X)                     # (B,)
        loss = criterion(logits, y.squeeze())
        if train:
            loss.backward()
            optimizer.step()
        preds = (torch.sigmoid(logits) > 0.5).int()
        correct += (preds == y.int().squeeze()).sum().item()
        total   += y.size(0)
        total_loss += loss.item() * y.size(0)
    return total_loss / total, correct / total


data_path = '/home/rayzhe0823/data/SEEDIV/Cross_data/'
files = sorted([f for f in os.listdir(data_path) if f.endswith('.mat')])
accs = []
for k, fname in enumerate(files):
    print(f"\n=== Subject {k}: {fname} ===")
    mat = sio.loadmat(os.path.join(data_path, fname))
    Xtr = mat['X_train']
    Xte = mat['X_test']
    ytr = mat['y_train'].ravel()
    yte = mat['y_test'].ravel()
    ytr[ytr == -1] = 0
    yte[yte == -1] = 0
    #N,T,C

    Xtr = Xtr.transpose(0, 2, 1)
    Xte = Xte.transpose(0, 2, 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xtr = torch.tensor(Xtr, dtype=torch.float32).permute(0, 1, 2)  # (N, C, T)
    ytr = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(Xte, dtype=torch.float32).permute(0, 1, 2)
    yte = torch.tensor(yte, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=64, shuffle=False)
    C = Xtr.shape[1]                       # 124
    model = DGCNN(in_channels=C, k=8).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ---------- 5. Train / Evaluate ----------


    for epoch in range(1, 201):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        test_loss,  test_acc  = run_epoch(test_loader,  train=False)
        if epoch % 10 == 0:
            print(f'E{epoch:03d} | '
                  f'Train: {train_acc*100:.1f}% / {train_loss:.3f}  | '
                  f'Test: {test_acc*100:.1f}% / {test_loss:.3f}')
    accs.append(test_acc)
print(accs)
print('mean:', np.mean(accs))
print('std:', np.std(accs))
print('std:', np.std(accs))

# [0.76, 0.6, 0.5, 0.54, 0.62, 0.5, 0.84, 0.5, 0.84, 0.58, 0.72, 0.58, 0.58, 0.74, 0.66, 0.54, 0.62, 0.58, 0.58, 0.64]
# mean: 0.6260000000000001
# std: 0.1016070863670443

# SEED:
# [0.5966666666666667, 0.5733333333333334, 0.67, 0.7533333333333333, 0.6533333333333333, 0.8466666666666667, 0.7766666666666666, 0.9333333333333333, 0.76, 0.65, 0.7866666666666666, 0.6733333333333333, 0.7466666666666667, 0.73, 0.7666666666666667]
# mean: 0.7277777777777779
# std: 0.09119508325494606