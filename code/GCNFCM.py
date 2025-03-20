from torch_geometric.utils import to_dense_adj
import FCMQ
import GCNAE
import evaluate
import numpy as np
import torch
from torch_geometric.datasets import PolBlogs
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
from bishe.EQ import extended_overlapping_modularity
from bishe.Omega import Omega
from bishe.onmi import onmi


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


def GCNFCM(data,num):
    # dataset = PolBlogs(root='./data/Polblogs')
    # data = dataset[0]
    
    # 使用新方法生成特征矩阵
    A = to_dense_adj(data.edge_index).squeeze()
    A = A.float()
    X = generate_features(data.edge_index)
    
    # 预处理步骤
    A_norm = GCNAE.pre_process(A)
    
    # 初始化GCNAE模型 此为小模型使用
    #encoder = GCNAE.GCNEncoder(input_dim=X.size(1), hidden_dim=256, output_dim=32)
    # 初始化GCNAE模型 此为中模型使用
    encoder = GCNAE.GCNEncoder(input_dim=X.size(1), hidden_dim1=256,hidden_dim2=128,output_dim=32)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    lambda_param = 0.1
    # 训练GCNAE模型
    for epoch in range(num):
        optimizer.zero_grad()
        z = encoder(X, A_norm)
        ipd_decoder = GCNAE.IPDDecoder(latent_dim=32)
        A_hat = ipd_decoder(z)
        # 此为小模型使用
        # gcd_decoder = GCNAE.GCDDecoder(input_dim=32, hidden_dim=256, output_dim=X.size(1))
        # 此为中模型使用
        gcd_decoder = GCNAE.GCDDecoder(input_dim=32, hidden_dim1=128, hidden_dim2=256,output_dim=X.size(1))
        X_hat = gcd_decoder(z, A_norm)
        loss = GCNAE.total_loss(A, A_hat, X, X_hat, lambda_param)
        # L = loss
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
    U_best, Q_hat = FCMQ.fcmq(Z, A, WE, epsilon=0.001, max_c=10, m=2, lamb=0.01)
    
    # print("Best Soft Assignment Matrix U_best:", U_best)
    threshold = 0.5
    y_pred = (U_best > threshold).astype(int).argmax(axis=1)
    
    # 真实的标签
    y_true = data.y.numpy().squeeze()
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be a 1D array, but got shape {y_true.shape}")
    # 计算 NMI，ARI，Q_hat
    evaluate.nmi(y_true, y_pred)
    evaluate.ari(y_true, y_pred)
    print(f'Modularity Computation(˜ Q): {Q_hat:.4f}')
    # 计算 Omega 指标
    # 将 y_true 和 y_pred 转换为社区划分的字典格式
    comms1 = {}
    for node, comm in enumerate(y_true):
        if comm not in comms1:
            comms1[comm] = []
        comms1[comm].append(node)
    
    comms2 = {}
    for node, comm in enumerate(y_pred):
        if comm not in comms2:
            comms2[comm] = []
        comms2[comm].append(node)
    
    omega = Omega(comms1, comms2)
    print(f'Omega Score: {omega.omega_score:.4f}')
    
    # 计算 ONMI 指标
    cover_true = [set(comm) for comm in comms1.values()]
    cover_pred = [set(comm) for comm in comms2.values()]
    all_nodes = set(range(len(y_true)))
    onmi_score = onmi(cover_pred, cover_true, all_nodes, variant="LFK")
    print(f'ONMI Score: {onmi_score:.4f}')
    
    # 计算 EQ 指标
    # 将 y_pred 转换为社区划分的格式
    # communities = []
    node_memberships = {i: 0 for i in range(len(y_pred))}
    community_dict = {}
    for node, comm in enumerate(y_pred):
        if comm not in community_dict:
            community_dict[comm] = []
        community_dict[comm].append(node)
    
    communities = list(community_dict.values())
    
    # 计算每个节点所属的社区数量
    for node in range(len(y_pred)):
        for comm in communities:
            if node in comm:
                node_memberships[node] += 1
    
    # 计算 EQ
    adj_matrix = A.numpy()
    eq_value = extended_overlapping_modularity(adj_matrix, communities, node_memberships)
    print(f'EQ Value: {eq_value:.4f}')
