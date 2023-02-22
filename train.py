import nni
from dataloader import *
from model import *
from trainer import *
import warnings
warnings.filterwarnings('ignore')


def train(config):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    tuner_params = nni.get_next_parameter()
    config = update_nni_params(config, tuner_params)

    train_loader, val_loader, test_loader, A = get_loaders(config)
    model = Model(config, config['model'])
    trainer = Trainer(model, config, train_loader, val_loader, test_loader, A)
    trainer.train()


def update_nni_params(config, tuner_params):
    for k, v in config['trainer'].items():
        if k in tuner_params.keys():
            config['trainer'][k] = tuner_params[k]

    for k, v in config['model'].items():
        if k in tuner_params.keys():
            config['model'][k] = tuner_params[k]

    return config


if __name__ == '__main__':
    conf = {
        'trainer': {
            'use_gpu': True,
            'gpu_id': '1',
            'num_epoch': 200,
            'lr': 0.0001,
            'loss_target': 'min',
            'early_stop': 7,
            'is_save': False,
            'contra_weight': 0
        },
        'data_loader': {
            'batch_size': 10,
            'shuffle': True,
            'num_workers': 4,
            'drop_last': True,
        },
        'model': {
            'model_name':'ISTGCN',#'ISTGCN', 'LSTM', 'ASTGCN', 'AGCRN', 'MLP'
            'time_embedding_size': 8,
            'link_embedding_size': 8,
            'node_embed': 32,
            'hidden_size_0': 32,
            'hidden_size_1': 32,
            'hidden_size_2': 32,
            'hidden_size_3': 32,
            'num_layer': 1
        }
    }
    train(conf)
