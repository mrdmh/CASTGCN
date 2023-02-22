import copy
import pickle

import nni
import torch
import pandas as pd
from loss import *


class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader, A):
        self.config = config
        self.args = config['trainer']

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = get_device(config)
        self.model = model.to(self.device)
        self.A = A

        self.is_save = config['trainer']['is_save']

        self.loss_fn = loss_fn_mse
        self.contra_loss_fn = loss_fn_contra
        self.contra_weight = config['trainer']['contra_weight']
        self.model_name = config['model']['model_name']

        self.lr = self.args['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.num_epoch = self.args['num_epoch']
        self.early_stop = EarlyStop(self.args['loss_target'], self.args['early_stop'])


        print()

    def train(self, is_test=False):
        epoch = 0
        loss_batch = []
        loss_epoch = []
        rmse_epoch = []
        mae_batch = []
        mape_batch = []
        for epoch in range(self.num_epoch):
            train_metrics = Metrics()
            losses, loss = self.train_epoch(self.train_loader, self.A, train_metrics, epoch, is_test)
            loss_batch += losses
            loss_epoch.append(loss)
            rmse_epoch.append(train_metrics.rmse.avg)
            mae_batch.append(train_metrics.mae.avg)
            mape_batch.append(train_metrics.mape.avg)

            val_metrics = Metrics()
            self.val_epoch(self.val_loader, self.A, val_metrics, is_test)

            print(f'Epoch {epoch}: ')
            print(f'Train\t\t' + train_metrics.to_str())
            print(f'Validation\t' + val_metrics.to_str())

            nni.report_intermediate_result(val_metrics.rmse.avg)

            if self.early_stop.check(val_metrics.rmse.avg):
                print(f'\nEarly stop at epoch{epoch}!')
                break

            if is_test and epoch == 1:
                break

        test_metrics = Metrics()
        self.val_epoch(self.test_loader, self.A, test_metrics, is_test)

        nni.report_final_result(test_metrics.rmse.avg)
        print(f'\n**** Finished! ****\nTest\t' + test_metrics.to_str())

        print(f"\nspatial module: {self.model.spatial_module}")
        print(f"temporal module: {self.model.temporal_module}")

        return

    def train_epoch(self, dataloader, A, metrics, epoch, is_test=False):
        self.model.train()
        batch_loss = []
        for batch_idx, data in enumerate(dataloader):
            mix_x, mix_y, fea, label = [d.to(self.device) for d in data]
            pred, embed = self.model(mix_x, fea, A)
            pred_loss = self.loss_fn(pred, label)
            if self.model_name == 'GASTGCN':
                contra_loss = self.contra_loss_fn(embed, mix_y)
                loss = pred_loss + self.contra_weight * contra_loss
            else:
                loss = pred_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics.calculate(pred, label)

            batch_loss.append(loss.cpu().detach().item())
            if is_test:
                break
        return batch_loss, np.mean(batch_loss)

    def val_epoch(self, dataloader, A, metrics, is_test=False):
        self.model.eval()
        for batch_idx, data in enumerate(dataloader):
            mix_x, mix_y, fea, label = [d.to(self.device) for d in data]
            pred, embed = self.model(mix_x, fea, A)
            metrics.calculate(pred, label) #[B, N]

            if is_test:
                break


class EarlyStop:
    """
    For training process, early stop strategy
    """
    def __init__(self, mode='min', patience=20):
        self.mode = mode
        self.patience = patience
        self.idx = 0
        self.best_value = -float('inf')
        self.best_idx = 0

    def check(self, x):
        if self.mode == 'min':
            x = -x
        self.idx += 1

        if x > self.best_value:
            self.best_value = x
            self.best_idx = copy.deepcopy(self.idx)

        if self.idx < 10:
            return False
        elif self.idx - self.best_idx > self.patience:
            return True
        else:
            return False

def get_device(config):
    if config['trainer']['use_gpu'] and torch.cuda.is_available():
        cuda_id = config['trainer']['gpu_id']
        if cuda_id != 'none':
            device = torch.device(f"cuda:{cuda_id}")
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    config['trainer']['device'] = device
    return device
