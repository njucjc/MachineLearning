import os

def load_data(path, delim=','):
    """
    输入：数据文件的路径以及每行数据的分割符，默认分隔符为','
    输出：数据列表和标签列表
    """
    data, labels = [], []
    with open(path, 'r') as fp:
        for line in fp.readlines():
            line = line.strip().split(delim)
            data.append(list(map(float, line[:-1])))
            labels.append(int(line[-1]))

        return data, labels

def make_path(path):
    """
    输入：要创建的路径
    """
    if not os.path.exists(path):
        os.makedirs(path)