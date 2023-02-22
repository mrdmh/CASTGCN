import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import platform


class MultiDataset(Dataset):
    def __init__(self, mix_X, mix_Y, fea, label,  dataset='train'):
        super(MultiDataset, self).__init__()
        self.dataset = dataset
        self.mix_X = mix_X
        self.mix_Y = mix_Y
        self.fea = fea
        self.label = label
        self.len = len(label)


    def __getitem__(self, index):
        return torch.from_numpy(self.mix_X[index]).float(), \
               torch.from_numpy(self.mix_Y[index]).float(),\
            torch.from_numpy(self.fea[index]).float(), \
               torch.from_numpy(self.label[index]).float()

    def __len__(self):
        return self.len

    def normalize(self):
        return


class MultiDataLoader(DataLoader):
    def __init__(self, config, mix_X, mix_Y, X, Y,  dataset='train'):
        self.args = config['data_loader']
        # self.data_use = self.args['data_use']
        self.dataset = MultiDataset(mix_X, mix_Y, X, Y,  dataset)

        self.batch_size = self.args['batch_size']
        self.num_workers = self.args['num_workers']
        if platform.system().lower() == 'windows':
            self.num_workers = 0

        is_shuffle = True if dataset == 'train' and self.args['shuffle'] else False
        # is_drop_last = True if dataset == 'train' and self.args['drop_last'] else False
        super().__init__(dataset=self.dataset,
                         batch_size=self.args['batch_size'],
                         num_workers=self.num_workers,
                         shuffle=is_shuffle)


def get_loaders(config):
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
    from mixup import main
    mix_x, mix_y, train_x, train_y, val_x, val_y, test_x, test_y, A = main(ws)

    return MultiDataLoader(config, train_x, train_y, mix_x, mix_y,  'train'), \
           MultiDataLoader(config, val_x, val_y, val_x, val_y,  'val'), \
           MultiDataLoader(config, test_x, test_y, test_x, test_y,'test'), A


if __name__ == '__main__':
    pass
