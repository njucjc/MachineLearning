import numpy as np

def svd_train(raw_data, k):
    """
    输入：原始数据矩阵raw_data、降维后的维度k
    输出：投影矩阵
    参考：https://blog.csdn.net/u012421852/article/details/80439403
    """
    u, s, vt = np.linalg.svd(raw_data)
    return vt[:k, :len(vt)].T

def svd_test(new_data, v):
    """
    输入：新数据和投影矩阵
    输出：降维后的数据
    """
    return new_data.dot(v)
