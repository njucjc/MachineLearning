import numpy as np

def svd_train(raw_data, k):
    """
    输入：原始数据矩阵raw_data、降维后的维度k
    输出：投影矩阵
    参考：https://blog.csdn.net/u012421852/article/details/80439403
    """
    # 计算均值向量
    mean_vecs = np.mean(raw_data, axis=0)
    # 样本中心化
    centralized_data = raw_data - mean_vecs
    u, s, vt = np.linalg.svd(centralized_data)

    return vt[:k, :len(vt)].T, mean_vecs

def svd_test(new_data, mean_vec, v):
    """
    输入：新数据和投影矩阵
    输出：降维后的数据
    """
    new_data = new_data - mean_vec
    return new_data.dot(v)
