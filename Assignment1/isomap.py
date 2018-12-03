import numpy as np

"""
参考1：https://blog.csdn.net/m0_37783096/article/details/79704621?utm_source=blogxgwz6
参考2：https://github.com/unknown-kid/ISOmap_and_MDS/
"""

def get_dis_matrix(data):
    """
    输入：原始矩阵
    输出：距离矩阵
    """
    res = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            res[i, j] = np.linalg.norm(data[i] - data[j])
    
    return res

def floyd(m,k=10):

	inf = np.max(m)*1000
	row, col=m.shape
	d1 = np.ones((row, row)) * inf

	m_idx = np.argsort(m, axis=1)
	for i in range(row):
		d1[i, m_idx[i,0:k+1]] = m[i, m_idx[i,0:k+1]]
	for k in range(row):
		for i in range(row):
			for j in range(row):
				if d1[i, k] + d1[k, j] < d1[i, j]:
					d1[i, j] = d1[i, k]+d1[k, j]
	return d1

def mds(m):
	row, col = m.shape
	d2 = np.square(m)

	di = np.sum(d2, axis=1) / row
	dj = np.sum(d2, axis=0) / row
	dij = np.sum(d2) / (row ** 2)
	b = np.zeros((row, row))
	for i in range(row):
		for j in range(col):
			b[i, j] = (dij + d2[i, j] - di[i] - dj[j]) / (-2)
	return b

def isomap(data, n=10, k=10):
    # """
    # 输入：原始矩阵，降维后的维度，k近邻
    # 输出：降维后的矩阵
    # """
	m = get_dis_matrix(data)
	m = floyd(m,k)
	b = mds(m)
	eig_vals, eig_vecs = np.linalg.eigh(b)
	eig_vals_idx = np.argsort(-eig_vals)
	eig_vals = eig_vals[eig_vals_idx]
	eig_vecs = eig_vecs[:,eig_vals_idx]
	eig_vals_z = np.diag(eig_vals[0:n])
	eig_vecs_z = eig_vecs[:,0:n]
	return np.dot(eig_vecs_z, np.sqrt(eig_vals_z))