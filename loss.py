import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from InfoNCE_loss import InfoNCE


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

        # print('accident_num: ', accident_nums)
        # print('i:', i)
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)[:k]).keys()))
    return topk_list


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
    return hit_sum / sum(accident_grids_nums) * 100


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


def loss_fn_mse(avg, label):
    return F.mse_loss(avg, label)


def loss_fn_contra(embed, label):
    """
    :param embed: [batch, N, hidden]
    :param label: [batch, N]
    :return: loss
    """

    loss_obj = InfoNCE()

    batch, N = label.shape
    embed = embed.permute(1, 0, 2)      # [N, batch, hidden]
    label = label.permute(1, 0)         # [N, batch]

    loss = 0
    for i in range(N):
        single_embed = embed[i]
        single_label = label[i]

        if (single_label > 0).sum() == 0:
            continue
        pos_embed = single_embed[single_label > 0]
        neg_embed = single_embed[single_label == 0]

        half_flag = len(pos_embed) // 2
        if half_flag == 0:
            query = pos_embed
            positive = pos_embed
        else:
            query = pos_embed[:half_flag]
            if len(pos_embed) % 2 == 0:
                positive = pos_embed[half_flag:]
            else:
                positive = pos_embed[half_flag:-1]

        loss += loss_obj(query=query,
                         positive_key=positive,
                         negative_keys=neg_embed)

    return loss

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


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics(object):
    def __init__(self):
        self.rmse = AverageMeter()
        self.mae = AverageMeter()
        self.mape = AverageMeter()
        self.Recall = AverageMeter()
        self.MAP = AverageMeter()

    def calculate(self, pred, label):
        for metric, record in zip(['rmse', 'mae', 'Recall', 'MAP'],
                                  [self.rmse, self.mae, self.Recall, self.MAP]):
            value = evaluate(pred, label, metric)
            record.update(value, len(label))

    def to_str(self):  # return a string for print
        _str_ = ''
        for metric, record in zip(['rmse', 'mae', 'Recall', 'MAP'],
                                  [self.rmse, self.mae, self.Recall, self.MAP]):
            if metric == 'mape':
                _str_ += f'{metric}: {round(record.avg * 100, 2)}%\t'
            else:
                _str_ += f'{metric}: {round(record.avg, 2)}\t'
        return _str_
