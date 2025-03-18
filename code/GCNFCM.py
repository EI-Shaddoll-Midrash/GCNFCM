from torch_geometric.utils import to_dense_adj
import FCMQ
import GCNAE
import evaluate
import numpy as np
import torch
from torch_geometric.datasets import PolBlogs


# 生成特征矩阵
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


# def generate_features(edge_index):
#     # 获取节点数量
#     num_nodes = int(edge_index.max()) + 1
#
#     # 初始化特征矩阵为零矩阵
#     features = torch.zeros((num_nodes, num_nodes))
#
#     # 计算每个节点的度（出度）
#     degrees = torch.bincount(edge_index[0], minlength=num_nodes)
#
#     # 使用稀疏矩阵来计算邻接矩阵
#     # 修正稀疏矩阵创建方式
#     values = torch.ones(edge_index.shape[1])
#     adj = torch.sparse_coo_tensor(edge_index, values, torch.Size([num_nodes, num_nodes]))
#
#     # 计算共同邻居矩阵
#     # 修正稀疏矩阵乘法
#     common_neighbors = torch.sparse.mm(adj, adj).to_dense()
#
#     # 应用公式计算特征值
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             common_neighbors_count = common_neighbors[i, j].item()
#             degree_sum = degrees[i] + degrees[j]
#             if degree_sum > 0:
#                 feature_value = (2 * common_neighbors_count) / (degree_sum + 1e-8)  # 添加小常数避免除零
#                 features[i, j] = feature_value
#                 features[j, i] = feature_value  # 对称填充
#
#     return features


def main():
    dataset = PolBlogs(root='./data/Polblogs')
    data = dataset[0]
    
    # 使用新方法生成特征矩阵
    A = to_dense_adj(data.edge_index).squeeze()
    A = A.float()
    X = generate_features(data.edge_index)
    
    # 预处理步骤
    A_norm = GCNAE.pre_process(A)
    
    # 初始化GCNAE模型
    encoder = GCNAE.GCNEncoder(input_dim=X.size(1), hidden_dim=256, output_dim=32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    lambda_param = 0.1
    num = 200
    # 训练GCNAE模型
    for epoch in range(num):
        optimizer.zero_grad()
        z = encoder(X, A_norm)
        ipd_decoder = GCNAE.IPDDecoder(latent_dim=32)
        A_hat = ipd_decoder(z)
        gcd_decoder = GCNAE.GCDDecoder(input_dim=32, hidden_dim=256, output_dim=X.size(1))
        X_hat = gcd_decoder(z, A_norm)
        loss = GCNAE.total_loss(A, A_hat, X, X_hat, lambda_param)
        L = loss
        loss.backward()
        optimizer.step()
        # 动态调整 lambda_param
        if epoch % 10 == 0:
            L1 = GCNAE.L1_loss(A, A_hat)
            L2 = GCNAE.L2_loss(X, X_hat)
            if L1 > L2:
                lambda_param *= 1.05  # 增加 L1 的权重
            else:
                lambda_param *= 0.95  # 减少 L1 的权重
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num}], Loss: {loss.item():.4f}, lambda_param: {lambda_param:.4f}')
    
    # 使用GCNAE的潜在表示进行FCMQ社区检测
    Z = z.detach().numpy()
    WE = np.ones(A.shape)  # 假设所有边的权重为1
    U_best, Q_hat = FCMQ.fcmq(Z, A, WE, epsilon=0.001, max_c=8, m=2, lamb=0.01)
    
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
