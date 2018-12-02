import numpy as np
import argparse, os
from utils import load_data, make_path
from pca import pca_train, pca_test
from knn import knn

parser = argparse.ArgumentParser(description='Dimensionality Reduction')

parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--train_data', type=str, default='sonar', help='train data source')
parser.add_argument('--test_data', type=str, default='sonar', help='test data source')
parser.add_argument('--alg', type=str, default='pca', help='pca/svd/isomap')
parser.add_argument('--output', type=str, default='output', help='result output dir')
parser.add_argument('--dim', type=int, default= 10, help='target dim')

args = parser.parse_args()
make_path(args.output)

train_data_path = os.path.join('data', args.train_data + '-train.txt')
test_data_path = os.path.join('data', args.test_data + '-test.txt')

def train():
    train_data, _ = load_data(train_data_path)
    if args.alg == 'pca':
        mean_vecs, eig_vecs = pca_train(np.array(train_data), args.dim)
        np.savez(os.path.join(args.output, 'pca' + str(args.dim) + '.npz'), mean_vecs=mean_vecs, eig_vecs=eig_vecs)


def test():
    test_data, test_labels = load_data(test_data_path)
    train_data, train_labels = load_data(train_data_path)

    if args.alg == 'pca':
        saved_data  = np.load(os.path.join(args.output, 'pca' + str(args.dim) + '.npz'))
        mean_vecs = saved_data['mean_vecs']
        eig_vecs = saved_data['eig_vecs']

        reduction_test_data = pca_test(np.array(test_data), mean_vecs, eig_vecs)
        reduction_train_data = pca_test(np.array(train_data), mean_vecs, eig_vecs)

    acc = eval(reduction_test_data, test_labels, reduction_train_data, train_labels)
    print('ACC = ' + str(acc))

  
def eval(test_data, test_data_label, train_data, train_data_label):
    predict_label = []
    for x in test_data:
        label =  knn(x, train_data, train_data_label, 1)
        predict_label.append(label)

    right = 0
    size = len(test_data_label)
    for i in range(size):
        if predict_label[i] == test_data_label[i]:
            right = right + 1
    return right / size





if args.mode == 'train':
    train()
elif args.mode == 'test':
    test()



