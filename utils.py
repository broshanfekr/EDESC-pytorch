from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import h5py
import scipy.io
from sklearn import preprocessing
from sklearn.metrics.cluster import _supervised
from scipy.optimize import linear_sum_assignment
import pickle


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        # make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float32')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y


def load_cifar10_clip(data_path):
    imgs, labels, X = load_var(data_path)
    cluster_num = len(np.unique(labels))
    labels = np.asarray(labels)

    x = X.reshape((X.shape[0], -1)).astype('float32')
    y = labels.reshape((labels.size,))

    return x, y


    
def LoadDatasetByName(dataset_name, data_path):
    if dataset_name == 'reuters':
        x, y = load_reuters()
    elif dataset_name == "cifar10":
        x, y = load_cifar10_clip(data_path)
    return x, y

class LoadDataset(Dataset):

    def __init__(self, dataset_name, data_path=None):
        self.x, self.y = LoadDatasetByName(dataset_name, data_path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(labels_true, labels_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    # value = _supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
