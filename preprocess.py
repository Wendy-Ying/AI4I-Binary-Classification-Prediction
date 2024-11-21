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

from sklearn.cluster import KMeans
import numpy as np

def cluster_undersample(X, y, ratio=0.5):
    """
    使用 K-Means 聚类对多数类样本进行欠采样，控制多数类样本占比。
    
    参数:
        X: 特征矩阵 (numpy array)
        y: 标签向量 (numpy array)
        ratio: 欠采样后多数类样本占总样本的比例 (0 < ratio <= 1)。
        
    返回:
        X_resampled: 欠采样后的特征矩阵
        y_resampled: 欠采样后的标签向量
    """
    # Step 1: 统计类别分布，确定多数类和少数类
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    minority_class = unique_classes[np.argmin(counts)]
    
    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]
    n_minority = len(X_minority)
    
    # Step 2: 计算欠采样后多数类样本的目标数量
    n_total = int(n_minority / (1 - ratio))  # 总样本数
    n_majority_target = n_total - n_minority  # 欠采样后的多数类样本数

    # Step 3: 使用 K-Means 聚类对多数类样本欠采样
    kmeans = KMeans(n_clusters=n_majority_target, random_state=42)
    kmeans.fit(X_majority)
    
    # Step 4: 从每个簇中选择一个代表样本（离簇中心最近的点）
    cluster_centers = kmeans.cluster_centers_
    X_representatives = []
    for center in cluster_centers:
        distances = np.linalg.norm(X_majority - center, axis=1)
        representative_idx = np.argmin(distances)
        X_representatives.append(X_majority[representative_idx])
    
    X_representatives = np.array(X_representatives)
    y_representatives = np.full(len(X_representatives), majority_class)
    
    # Step 5: 合并少数类和代表多数类样本
    X_resampled = np.vstack((X_minority, X_representatives))
    y_resampled = np.hstack((np.full(n_minority, minority_class), y_representatives))
    
    return X_resampled, y_resampled
