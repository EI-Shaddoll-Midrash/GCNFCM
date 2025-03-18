import torch
import torch.nn as nn
import torch.nn.functional as F


# 预处理步骤
def pre_process(A):
    A_with_self_loops = A + torch.eye(A.size(0), device=A.device)
    
    # 计算度矩阵
    D = torch.sum(A_with_self_loops, dim=1)
    
    # 计算度矩阵的逆平方根
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    
    # 计算归一化邻接矩阵
    A_norm = torch.matmul(torch.matmul(D_inv_sqrt, A_with_self_loops), D_inv_sqrt)
    # print("Normalized Adjacency Matrix:\n", A_norm)
    return A_norm


# GCN编码器
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, A_norm):
        # 第一层卷积操作，使用ReLU激活函数
        h = F.relu(self.conv1(A_norm @ x))
        # 第二层卷积操作，使用线性激活函数（即不使用激活函数）
        z = self.conv2(h)
        return z


# IPD解码器
class IPDDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(IPDDecoder, self).__init__()
        self.latent_dim = latent_dim
    
    def forward(self, z):
        # 计算Z的转置与Z的内积
        ZZT = torch.matmul(z, z.t())
        # 应用sigmoid函数作为激活函数
        A_hat = torch.sigmoid(ZZT)
        return A_hat


# GCD解码器
class GCDDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCDDecoder, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, A_norm):
        # 应用图卷积操作
        h = F.relu(self.gc1(A_norm @ z))
        X_hat = self.gc2(h)
        return X_hat


# 损失函数
def L1_loss(A, A_hat):
    criterion = nn.MSELoss()
    loss = criterion(A_hat, A)
    
    return loss


def L2_loss(X, X_hat):
    criterion = nn.MSELoss()
    loss = criterion(X_hat, X)
    return loss


def total_loss(A, A_hat, X, X_hat, lambda_param):
    L1 = L1_loss(A, A_hat)
    L2 = L2_loss(X, X_hat)
    L = L1 + lambda_param * L2
    return L
