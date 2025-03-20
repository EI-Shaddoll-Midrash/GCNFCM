import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.datasets import PolBlogs
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import scipy.io as sio
from bishe.GCNFCM import GCNFCM
import torch


# 读取gml格式并转化
def convert_polbooks_to_pyg_data(gml_file_path):
    # 读取GML文件
    G = nx.read_gml(gml_file_path)
    
    # 提取节点特征
    node_features = []
    node_labels = []
    for node in G.nodes(data=True):
        node_labels.append(node[0])  # 节点的名称
        if node[1]['value'] == 'n':
            node_features.append([1, 0, 0])  # 中立
        elif node[1]['value'] == 'c':
            node_features.append([0, 1, 0])  # 保守派
        elif node[1]['value'] == 'l':
            node_features.append([0, 0, 1])  # 自由派
    
    # 创建节点名称到索引的映射
    node_index_map = {name: idx for idx, name in enumerate(node_labels)}
    
    # 提取边信息
    edge_index = []
    for edge in G.edges():
        source = node_index_map[edge[0]]
        target = node_index_map[edge[1]]
        edge_index.append([source, target])
        edge_index.append([target, source])  # 无向图需要双向边
    
    # 提取节点标签
    y = []
    for node in G.nodes(data=True):
        if node[1]['value'] == 'n':
            y.append(0)  # 中立
        elif node[1]['value'] == 'c':
            y.append(1)  # 保守派
        elif node[1]['value'] == 'l':
            y.append(2)  # 自由派
    
    # 转换为PyTorch张量
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor(y, dtype=torch.long)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


# 读取mat对象
def convert_acm_to_pyg_data(mat_file_path):
    # 读取 .mat 文件
    mat_data = sio.loadmat(mat_file_path)
    
    # 提取数据
    W = mat_data['W']  # 邻接矩阵
    fea = mat_data['fea']  # 节点特征
    gnd = mat_data['gnd']  # 节点标签
    
    # 提取边索引
    row, col = W.nonzero()  # 获取非零元素的行和列索引
    edge_index = torch.tensor([row, col], dtype=torch.long)  # 转换为形状为 [2, num_edges] 的张量
    
    # 节点特征和标签
    x = torch.tensor(fea.todense(), dtype=torch.float)  # 节点特征
    y = torch.tensor(gnd[0], dtype=torch.long)  # 节点标签
    
    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def convert_wiki_to_pyg_data(mat_file_path):
    # 读取 .mat 文件
    mat_data = sio.loadmat(mat_file_path)
    
    # 提取数据
    W = mat_data['W']  # 邻接矩阵
    fea = mat_data['fea']  # 节点特征
    gnd = mat_data['gnd']  # 节点标签
    
    # 提取边索引
    row, col = W.nonzero()  # 获取非零元素的行和列索引
    edge_index = torch.tensor([row, col], dtype=torch.long)  # 转换为形状为 [2, num_edges] 的张量
    
    # 节点特征和标签
    x = torch.tensor(fea.todense(), dtype=torch.float)  # 节点特征
    y = torch.tensor(gnd, dtype=torch.long)  # 节点标签
    
    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def main():
    # # Polblogs数据集
    # dataset = PolBlogs(root='./data/Polblogs')
    # data = dataset[0]
    # GCNFCM(data, 200)
    # print("=============================================")
    # # 加载 Karate Club 数据集
    # dataset = KarateClub()
    # data = dataset[0]  # 数据集中只有一个图
    # GCNFCM(data, 1000)
    # print("==============================================")
    #
    # # 加载 Football 数据集  正在尝试
    # # gml_file_path = r'D:\pythonProject\bishe\data\Football\football.gml'
    # # # 使用自定义函数读取 GML 文件
    # # G = read_gml_with_multiedges(gml_file_path)
    # # # 转换为 PyTorch Geometric 的 Data 对象
    # # data = convert_multigraph_to_pyg_data(G)
    # # # 打印调试信息，检查 edge_index 是否正确
    # # print("Edge Index Shape:", data.edge_index.shape)
    # # print("Number of Nodes:", data.num_nodes)
    # # print("Number of Edges:", data.num_edges)
    # # GCNFCM(data, 1000)
    #
    # # 加载 Polbooks 数据集
    # gml_file_path = r'D:\pythonProject\bishe\data\Polbooks\polbooks.gml'
    # # 转换为 PyTorch Geometric 的 Data 对象
    # data = convert_polbooks_to_pyg_data(gml_file_path)
    # # 打印调试信息，检查 edge_index 是否正确
    # # print("Edge Index Shape:", data.edge_index.shape)
    # # print("Number of Nodes:", data.num_nodes)
    # # print("Number of Edges:", data.num_edges)
    # GCNFCM(data, 1000)
    # print("===========================================================")
    
    # acm数据集
    mat_file_path = r"D:\pythonProject\bishe\data\Acm\acm.mat"
    data = convert_acm_to_pyg_data(mat_file_path)
    GCNFCM(data, 500)
    print("================================================")
    # wiki
    mat_file_path = r"D:\pythonProject\bishe\data\Wiki\wiki.mat"
    data = convert_acm_to_pyg_data(mat_file_path)
    GCNFCM(data, 500)


if __name__ == "__main__":
    main()
