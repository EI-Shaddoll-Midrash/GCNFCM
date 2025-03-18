import numpy as np


def initialize_community_centers(Z, c):
    # 固定随机种子
    np.random.seed(42)
    # 随机初始化社区中心
    n, d = Z.shape
    indices = np.random.choice(n, c, replace=False)
    return Z[indices]


# def update_soft_assignment(Z, C, m):
#     # n, d = Z.shape
#     # c = C.shape[0]
#
#     # 计算所有数据点到所有聚类中心的距离
#     distances = np.linalg.norm(Z[:, np.newaxis, :] - C, axis=2)
#
#     # 计算距离比值的幂次方
#     ratio = np.power(distances, 2 / (m - 1))
#
#     # 计算每个数据点到每个聚类中心的比值
#     ratio_matrix = np.exp(-ratio)
#
#     # 计算软分配矩阵 U
#     sum_ratio = np.sum(ratio_matrix, axis=1, keepdims=True)
#     sum_ratio = np.where(sum_ratio == 0, 1, sum_ratio)  # 避免除以零
#     U = ratio_matrix / sum_ratio
#
#     return U
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

