import numpy as np

def pca_train(raw_data, k):
    """
    输入：原始数据矩阵raw_data、降维后的维度k
    输出：样本均值和投影向量
    参考：https://www.cnblogs.com/lzllovesyl/p/5235137.html
    """
    # 计算均值向量
    mean_vecs = np.mean(raw_data, axis=0)
    # 样本中心化
    centralized_data = raw_data - mean_vecs
    # 计算协方差矩阵
    cov_mat = np.cov(centralized_data, rowvar=False)
    # 计算特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_mat))
    # 特征值排序
    eig_vals_idx = np.argsort(eig_vals)
    topk_idx = eig_vals_idx[:-(k+1):-1]
    # 取对应的特征向量
    topk_eig_vecs= eig_vecs[:,topk_idx]
    return mean_vecs, topk_eig_vecs

def pca_test(new_data, mean_vecs, eig_vecs):
    """
    输入：新样本new_data、均值向量mean_vec、和投影向量eig_vec
    输出：投影后的数据
    """
    # 数据中心化
    centralized_data = new_data - mean_vecs
    # 降维
    return centralized_data.dot(eig_vecs)




