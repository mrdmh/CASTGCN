import numpy as np
import os

def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file

def main(ws):
    data = np.load(ws + '/GASTGCN/data/generaged_data.npy', allow_pickle=True).item()
    fea = data['fea']
    label = data['label']
    A = data['A']
    train_index = int(0.6 * len(fea))
    val_index = int(0.8 * len(fea))
    train_x, train_y = fea[:train_index], label[:train_index]
    val_x, val_y = fea[train_index:val_index], label[train_index:val_index]
    test_x, test_y = fea[val_index:], label[val_index:]

    mix_fea_l = []
    mix_label_l = []
    lamd = 0.1
    for j in range(len(train_y) - 7):
        fea_j = train_x[j]
        fea_i = train_x[j + 7]
        mix_fea = lamd * fea_i + (1 - lamd) * fea_j
        label_j = train_y[j]
        label_i = train_y[j + 7]
        mix_label = lamd * label_i + (1 - lamd) * label_j
        mix_fea_l.append(mix_fea)
        mix_label_l.append(mix_label)
    X_mix = np.array(mix_fea_l)
    Y_mix = np.array(mix_label_l)

    return X_mix, Y_mix, train_x, train_y, val_x, val_y, test_x, test_y, A


if __name__ == "__main__":
    ws = get_workspace()
    main(ws)
