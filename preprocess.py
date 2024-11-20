import numpy as np
from sklearn.neighbors import KDTree

def adasyn(X, y, minority_class, beta=1.0, k=5):
    """
    使用 sklearn 的 KDTree 实现 ADASYN 算法。
    
    参数:
        X: 特征矩阵 (numpy array)
        y: 标签向量 (numpy array)
        minority_class: 少数类的标签值
        beta: 平衡比参数，控制生成样本数量
        k: k-近邻中的邻居数量
    返回:
        X_resampled: 平衡后的特征矩阵
        y_resampled: 平衡后的标签向量
    """
    # Step 1: 划分多数类和少数类
    X_minority = X[y == minority_class]
    X_majority = X[y != minority_class]
    n_minority = len(X_minority)
    n_majority = len(X_majority)

    # 需要生成的样本总数
    G = int((n_majority - n_minority) * beta)

    # Step 2: 使用 KDTree 计算每个少数类样本的 k-近邻
    tree = KDTree(X, leaf_size=30)  # 构建 KD 树
    difficulty = np.zeros(n_minority)
    
    for i, x in enumerate(X_minority):
        # 获取 k-近邻（包括多数类和少数类）
        dist, ind = tree.query([x], k=k+1)  # 包括自身，取 k+1 个
        neighbor_indices = ind[0][1:]  # 排除自身
        neighbor_labels = y[neighbor_indices]
        # 统计邻居中多数类的比例
        majority_count = np.sum(neighbor_labels != minority_class)
        difficulty[i] = majority_count / k

    # 正规化难度系数
    difficulty_normalized = difficulty / np.sum(difficulty)

    # 每个样本需要生成的样本数
    g_i = np.round(difficulty_normalized * G).astype(int)

    # Step 3: 生成合成样本
    X_synthetic = []
    for i, x in enumerate(X_minority):
        for _ in range(g_i[i]):
            # 从 k 个近邻中随机选择一个邻居
            dist, ind = tree.query([x], k=k+1)
            neighbor_idx = np.random.choice(ind[0][1:])  # 排除自身
            neighbor = X[neighbor_idx]
            # 沿着 x 和邻居生成新样本
            diff = neighbor - x
            new_sample = x + np.random.rand() * diff
            X_synthetic.append(new_sample)

    # 拼接原始数据和生成的样本
    X_resampled = np.vstack((X, np.array(X_synthetic)))
    y_resampled = np.hstack((y, np.full(len(X_synthetic), minority_class)))

    return X_resampled, y_resampled