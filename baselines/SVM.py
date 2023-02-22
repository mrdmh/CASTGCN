import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
import os


def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


ws = get_workspace()
def Recall(y_true, y_pred):
    """
    Arguments:
        y_true {np.ndarray} -- shape (samples, num_link)
        y_pred {np.ndarray} -- shape (samples, num_link)
    Returns:
        float -- recall
    """

    accident_grids_nums = nonzero_num(y_true)

    true_top_k = get_top(y_true, accident_grids_nums)
    pred_top_k = get_top(y_pred, accident_grids_nums)

    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums)


def AP(label_list, pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0

def nonzero_num(y_true):
    """get the grid number of have traffic accident in all time interval

    Arguments:
        y_true {np.array} -- shape:(samples,pre_len,W,H)
    Returns:
        {list} -- (samples,)
    """
    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list


def get_top(data, accident_nums):
    """get top-K risk grid
    Arguments:
        data {np.array} -- shape (samples, num_link)
        accident_nums {list} -- (samples,)，grid number of have traffic accident in all time intervals
    Returns:
        {list} -- (samples,k)
    """
    data = data.reshape((data.shape[0], -1))
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)[:k]).keys()))
    return topk_list

def MAP(y_true, y_pred):
    """
        y_true {np.ndarray} -- shape (samples, num_link)
        y_pred {np.ndarray} -- shape (samples, num_link)
    """

    accident_grids_nums = nonzero_num(y_true)

    true_top_k = get_top(y_true, accident_grids_nums)
    pred_top_k = get_top(y_pred, accident_grids_nums)

    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]), list(pred_top_k[sample])))
    return sum(all_k_AP) / len(all_k_AP)

def evaluate(pred, label, metric='mse'):
    if metric == 'mae':
        result = nn.L1Loss()(pred, label).item()
    elif metric == 'mse':
        result = nn.MSELoss()(pred, label).item()
    elif metric == 'rmse':
        mse = nn.MSELoss()(pred, label).item()
        result = np.sqrt(mse)
    elif metric == 'mape':
        result = torch.abs((pred - label) / label).mean().item()
    elif metric == 'Recall':
        result = Recall(label, pred).item()
    elif metric == 'MAP':
        result = MAP(label, pred)
    else:
        print('Metric error')
        return
    return result

data = np.load(ws + '/data/generaged_data.npy', allow_pickle=True).item()
fea = data['fea']
label = data['label']
A = data['A']

train_index = int(0.6 * len(fea))
val_index = int(0.8 * len(fea))
train_x, train_y = fea[:train_index], label[:train_index]
val_x, val_y = fea[train_index:val_index], label[train_index:val_index]
test_x, test_y = fea[val_index:], label[val_index:]
svm_model = SVC(kernel='rbf')

train_num = train_x.shape[0]
time_lag = train_x.shape[1]
node_num = train_x.shape[2]
num_fea = train_x.shape[3]
test_num = test_x.shape[0]
svm_model.fit(np.transpose(train_x, (0, 2, 1, 3)).reshape(train_num * node_num, time_lag * num_fea), train_y.reshape(train_num * node_num, 1))
prediction = svm_model.predict(np.transpose(test_x, (0, 2, 1, 3)).reshape(test_num * node_num, time_lag * num_fea))
prediction = torch.from_numpy(prediction).reshape(test_num, node_num).float()
test_label = torch.from_numpy(test_y).reshape(test_num, node_num).float()
print('MAE: ', evaluate(prediction, test_label, 'mae'))
print('MAP: ', evaluate(prediction, test_label, 'MAP'))
print('Recall: ', evaluate(prediction, test_label, 'Recall'))



