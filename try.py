from torch_geometric.utils import to_dense_adj
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import numpy as np
import torch
from torch_geometric.datasets import PolBlogs
def generate_features(edge_index):
    # 获取节点数量
    num_nodes = int(edge_index.max()) + 1
    
    # 初始化特征矩阵为零矩阵
    features = torch.zeros((num_nodes, num_nodes))
    
    # 计算每个节点的度（出度）
    degrees = torch.bincount(edge_index[0], minlength=num_nodes)
    
    # 使用稀疏矩阵来计算邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    
    # 计算共同邻居矩阵
    common_neighbors = torch.matmul(adj, adj)
    
    # 应用公式计算特征值
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            common_neighbors_count = common_neighbors[i, j].item()
            degree_sum = degrees[i] + degrees[j]
            if degree_sum > 0:
                feature_value = (2 * common_neighbors_count) / degree_sum
                features[i, j] = feature_value
                features[j, i] = feature_value  # 对称填充
    
    return features


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


def update_soft_assignment(Z, C, m):
    n, d = Z.shape
    c = C.shape[0]
    
    # 计算所有数据点到所有聚类中心的距离
    distances = np.linalg.norm(Z[:, np.newaxis, :] - C, axis=2)
    distances = np.where(distances == 0, 1e10, distances)
    # 计算软分配矩阵 U
    U = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            denominator = np.sum((distances[i, j] / distances[i, :]) ** (2 / (m - 1)))
            U[i, j] = 1 / denominator
    
    return U


def update_community_centers(Z, U, m):
    n, d = Z.shape
    c = U.shape[1]
    
    # 初始化聚类中心数组
    C = np.zeros((c, d))
    
    # 更新聚类中心
    for j in range(c):
        # 扩展 U[:, j] ** m 为 (n, 1) 以实现与 Z 的广播
        Uj_m = (U[:, j] ** m).reshape(-1, 1)
        C[j] = np.sum(Uj_m * Z, axis=0) / np.sum(Uj_m)
    
    return C

def initialize_community_centers(Z, c):
    # 固定随机种子
    np.random.seed(42)
    # 随机初始化社区中心
    n, d = Z.shape
    indices = np.random.choice(n, c, replace=False)
    return Z[indices]


def calculate_modularity(U, A, WE, lamb, c):
    n = A.shape[0]
    # print("n:", n)
    # print("c:", c)
    # print("U.shape:", U.shape)
    
    m = np.sum(WE) / 2  # 总边权和
    Q = 0.0
    
    for k in range(c):
        Vk = np.where(U[:, k] > lamb)[0]
        U_Vk = U[Vk, k]
        U_Vk_complement = 1 - U[np.setdiff1d(np.arange(n), Vk), k]
        A_Vk_Vk = np.sum(U_Vk[:, np.newaxis] * U_Vk * WE[Vk[:, np.newaxis], Vk])
        A_Vk_V = np.sum(
            U_Vk[:, np.newaxis] * U_Vk_complement * WE[Vk[:, np.newaxis], np.setdiff1d(np.arange(n), Vk)]) / 2
        
        Q += (A_Vk_Vk / m - (A_Vk_V / m) ** 2)
    # print(Q)
    return Q

def fcmq(Z, A, WE, epsilon, max_c, m, lamb):
    n, _ = Z.shape
    best_U = None
    prev_Q = -np.inf
    c = 2  # 初始社区数量
    U = np.ones((n, c)) / c  # 初始软分配矩阵
    
    while c <= max_c:
        C = initialize_community_centers(Z, c)
        U = update_soft_assignment(Z, C, m)
        
        while True:
            
            U_prev = U
            U = update_soft_assignment(Z, C, m)
            C = update_community_centers(Z, U, m)
            if np.linalg.norm(U - U_prev) < epsilon:
                break
        
        Q = calculate_modularity(U, A, WE, lamb, c)  # 假设已实现模块度计算
        if Q > prev_Q:
            best_U = U
            prev_Q = Q
        c += 1  # 增加社区数量
        # print(U)
        # print(c)
    best_Q = Q
    return best_U, best_Q

def main():
    dataset = PolBlogs(root='./data/Polblogs')
    data = dataset[0]
    
    # 使用新方法生成特征矩阵
    A = to_dense_adj(data.edge_index).squeeze()
    A = A.float()
    X = generate_features(data.edge_index)
    
    # 预处理步骤
    A_norm = pre_process(A)
    
    # 初始化GCNAE模型
    encoder = GCNEncoder(input_dim=X.size(1), hidden_dim=256, output_dim=32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    # 训练GCNAE模型
    for epoch in range(200):
        optimizer.zero_grad()
        z = encoder(X, A_norm)
        ipd_decoder = IPDDecoder(latent_dim=32)
        A_hat = ipd_decoder(z)
        gcd_decoder = GCDDecoder(input_dim=32, hidden_dim=256, output_dim=X.size(1))
        X_hat = gcd_decoder(z, A_norm)
        loss = total_loss(A, A_hat, X, X_hat, 0.1)
        L = loss
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}')
    
    # 使用GCNAE的潜在表示进行FCMQ社区检测
    Z = z.detach().numpy()
    WE = np.ones(A.shape)  # 假设所有边的权重为1
    U_best, Q_hat = fcmq(Z, A, WE, epsilon=0.001, max_c=8, m=2,lamb=0.01)
    
    # print("Best Soft Assignment Matrix U_best:", U_best)
    threshold = 0.5
    y_pred = (U_best > threshold).astype(int).argmax(axis=1)
    
    # 真实的标签
    y_true = data.y.numpy()
    
    # 计算 NMI，ARI，Q_hat
    evaluate.nmi(y_true, y_pred)
    evaluate.ari(y_true, y_pred)
    print(f'Modularity Computation(˜ Q): {Q_hat:.4f}')


if __name__ == "__main__":
    main()
