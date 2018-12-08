import numpy as np

def knn(x, data, labels, k):
    """
    输入：新样本x，训练集data，相关标签labels，k值
    输出：k近邻下x的标签
    参考：https://www.cnblogs.com/ybjourney/p/4702562.html
    """
    # 计算数据共有几行
    line_count = data.shape[0] 

    # 计算差值矩阵
    diff = np.tile(x, (line_count, 1)) - data
    # 差值求平方
    squared_diff = diff ** 2 

    # 行加和（距离的平方）
    squared_dist = np.sum(squared_diff, axis = 1)

    # 求欧式距离
    distance = squared_dist ** 0.5
    # 距离排序
    sorted_dist_idx = np.argsort(distance)
    count = {} 
    for i in range(k):
        vote_label = labels[sorted_dist_idx[i]]
        count[vote_label] = count.get(vote_label, 0) + 1

    return max(count, key=count.get)
    
