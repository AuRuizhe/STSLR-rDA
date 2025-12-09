# python/models/dgcnn.py
"""
DGCNN model for binary classification on EEG-like time-series data.

Input shape: (B, C, T), where
    B = batch size
    C = number of channels (features)
    T = number of time points (treated as "nodes")

This implementation follows the standard DGCNN design:
    - k-NN graph in feature space
    - EdgeConv blocks
    - Global pooling + MLP for classification
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest neighbors based on pairwise Euclidean distance.

    Args:
        x: Tensor of shape (B, F, N), where N is the number of nodes.
        k: Number of neighbors.

    Returns:
        idx: LongTensor of shape (B, N, k) containing neighbor indices.
    """
    # x: (B, F, N) --> x_T: (B, N, F)
    # pairwise distance (negative) so that topk gives nearest neighbors
    inner = -2 * torch.matmul(x.transpose(2, 1), x)         # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)             # (B, 1, N)
    dist = -xx - inner - xx.transpose(2, 1)                 # (B, N, N)
    idx = dist.topk(k=k, dim=-1)[1]                         # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Construct edge features for each node and its k neighbors.

    Args:
        x: Tensor of shape (B, F, N).
        k: Number of neighbors.

    Returns:
        feature: Tensor of shape (B, 2F, N, k), concatenating
                 (neighbor - center, center) along the channel dimension.
    """
    B, F, N = x.size()
    idx = knn(x, k=k)                                       # (B, N, k)

    # flatten batch and node dims for indexing
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base                                    # (B, N, k)
    idx = idx.view(-1)                                      # (B*N*k,)

    # (B, F, N) -> (B*N, F)
    x_flat = x.transpose(2, 1).contiguous().view(B * N, F)
    neighbor = x_flat[idx, :].view(B, N, k, F)              # (B, N, k, F)
    neighbor = neighbor.permute(0, 3, 1, 2).contiguous()    # (B, F, N, k)

    x_center = x.view(B, F, N, 1).repeat(1, 1, 1, k)        # (B, F, N, k)
    feature = torch.cat((neighbor - x_center, x_center), dim=1)  # (B, 2F, N, k)
    return feature


class EdgeConv(nn.Module):
    """
    EdgeConv block used in DGCNN.

    Input:
        x: (B, F, N)

    Output:
        out: (B, out_channels, N)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 k: int = 8, use_bn: bool = True) -> None:
        super().__init__()
        self.k = k

        layers = [
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, N)
        feat = get_graph_feature(x, k=self.k)   # (B, 2F, N, k)
        out = self.mlp(feat)                   # (B, out_channels, N, k)
        out = out.max(dim=-1)[0]               # max over neighbors -> (B, out_channels, N)
        return out


class DGCNN(nn.Module):
    """
    DGCNN for binary classification.

    Args:
        in_channels: Number of input feature channels (e.g., EEG channels).
        k: Number of neighbors in k-NN graph.
        emb_dims: Embedding dimension of the penultimate layer.
        dropout: Dropout rate in the MLP head.

    Forward:
        x: (B, C, T)  # C=in_channels, T=#nodes
        returns: logits of shape (B,)
    """

    def __init__(self,
                 in_channels: int,
                 k: int = 8,
                 emb_dims: int = 256,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.ec1 = EdgeConv(in_channels, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)

        self.linear1 = nn.Linear(64 + 64 + 128, emb_dims, bias=False)
        self.bn0 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(emb_dims, 64)
        self.linear3 = nn.Linear(64, 1)  # binary logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x1 = self.ec1(x)                 # (B, 64, T)
        x2 = self.ec2(x1)                # (B, 64, T)
        x3 = self.ec3(x2)                # (B, 128, T)

        # global max pooling over nodes for each EdgeConv output
        x1_pool = x1.max(dim=-1)[0]      # (B, 64)
        x2_pool = x2.max(dim=-1)[0]      # (B, 64)
        x3_pool = x3.max(dim=-1)[0]      # (B, 128)

        x_cat = torch.cat((x1_pool, x2_pool, x3_pool), dim=1)  # (B, 256)

        x = self.linear1(x_cat)
        x = self.bn0(x)
        x = F.leaky_relu(x)
        x = self.dp1(x)
        x = F.leaky_relu(self.linear2(x))
        logits = self.linear3(x).squeeze(1)  # (B,)
        return logits


if __name__ == "__main__":
    # Quick sanity check with random input
    B, C, T = 8, 62, 1000
    model = DGCNN(in_channels=C, k=8)
    x = torch.randn(B, C, T)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)   # expect: (B,)
