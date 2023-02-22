import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from baselines.ASTGCN import ASTGCN_submodule
from baselines.AGCRN import AGCRN
from baselines.ISTGCN import ISTGCN

class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class TCN_block(nn.Module):
    def __init__(self, in_fea, hidden, num_node, time_steps, device='cpu'):
        super(TCN_block, self).__init__()
        self.num_node = num_node
        self.TAttn = Temporal_Attention_layer(device, in_fea, num_node, time_steps)
        self.temporal_conv = nn.Conv2d(in_channels=in_fea,
                                       out_channels=hidden,
                                       kernel_size=(1, 3),
                                       stride=(1, 1),
                                       padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels=in_fea,
                                       out_channels=hidden,
                                       kernel_size=(1, 1),
                                       stride=(1, 1))
        self.ln = nn.LayerNorm(hidden)
        self.time_conv = nn.Sequential(
            # [B*N, hidden, 6] -> [B*N, hidden, 3]
            nn.Conv1d(in_channels=hidden,
                      out_channels=hidden,
                      kernel_size=4,
                      stride=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            # [B*N, hidden, 3] -> [B*N, hidden, 1]
            nn.Conv1d(in_channels=hidden,
                      out_channels=hidden,
                      kernel_size=3,
                      stride=1),
        )

    def forward(self, x):
        # x: (B, N, T, in_fea)
        batch, num_node, time_steps, in_fea = x.shape
        # t_a: [B, T, T]
        t_a = self.TAttn(x.permute(0, 1, 3, 2))
        # x_TAt: [B, N, in_fea, T]
        x_TAt = torch.matmul(x.reshape(batch, -1, time_steps), t_a).reshape(batch,
                                                                            num_node,
                                                                            in_fea,
                                                                            time_steps)
        # time_conv_output: [B, hidden, N, T]
        time_conv_output = self.temporal_conv(x_TAt.permute(0, 2, 1, 3))
        # x_residual: [B, hidden, N, T]
        x_residual = self.residual_conv(x.permute(0, 3, 1, 2))
        # layer_norm: [B, T, N, hidden]
        # output: [B, N, hidden, T]
        output = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # output: [B, N, hidden, T] -> [B*N, hidden, T] -> [B*N, hidden, 1]
        output = output.reshape(batch * num_node, -1, time_steps)
        output = self.time_conv(output)
        # return: [B*N, hidden, 1] -> [B*N, hidden]-> [B, N, hidden]
        output = output.squeeze().reshape(batch, num_node, -1)
        return output


class AdaptiveGraphConv(nn.Module):
    def __init__(self, feature_dim, hidden_size, node_num=5, embedding_size=128):
        super(AdaptiveGraphConv, self).__init__()
        self.node_embedding = nn.Parameter(torch.Tensor(node_num, embedding_size))
        self.weights = nn.Parameter(torch.FloatTensor(embedding_size, feature_dim, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(embedding_size, hidden_size))

        # self.conv_spatial = nn.Linear(feature_dim, hidden_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        # x: [B, T, N, hidden]
        adj_spatial = torch.eye(self.node_embedding.shape[0]).to(self.node_embedding.device) + \
                      F.softmax(F.relu(torch.mm(self.node_embedding,
                                                self.node_embedding.transpose(0, 1))), dim=1)

        # W: d * C * Hidden
        e_weights = torch.einsum('nd,dcf->ncf', self.node_embedding, self.weights)
        # b: d * Hidden
        e_bias = torch.einsum('nd,df->nf', self.node_embedding, self.bias)
        z_x = torch.einsum('nn, btnc->btnc', adj_spatial, x)
        y = torch.einsum('btnc, ncf->btnf', z_x, e_weights) + e_bias
        return y


class Model(nn.Module):
    def __init__(self, config, params):
        super(Model, self).__init__()
        self.config = config
        self.params = params
        self.device = get_device(config)
        print(f'device: {self.device}')

        self.num_node = 36
        self.input_hidden = params['hidden_size_0']
        self.cat_embed_dim = 8
        self.time_lag = 10
        self.temporal_hidden = params['hidden_size_1']
        self.spatial_hidden = params['hidden_size_2']
        self.node_embed_size = params['node_embed']
        self.output_hidden = params['hidden_size_3']
        self.model = params['model_name']
        self.temporal_num_layer = params['num_layer']
        self.fea_dim = 8
        self.agcrn = AdaptiveGraphConv(feature_dim=self.input_hidden + params['time_embedding_size'],
                                                   hidden_size=self.spatial_hidden,
                                                   node_num=self.num_node,
                                                   embedding_size=self.node_embed_size)
        self.adap_s =  AdaptiveGraphConv(feature_dim=self.input_hidden + params['time_embedding_size'],
                                                   hidden_size=self.spatial_hidden,
                                                   node_num=self.num_node,
                                                   embedding_size=self.node_embed_size)

        self.slice_embedding = nn.Embedding(7, params['time_embedding_size'])
        self.astgcn_output = nn.Linear(params['hidden_size_0'], 1)

        self.input_layer = nn.Linear(7, self.input_hidden)
        self.istgcn = ISTGCN(self.num_node, self.fea_dim, self.time_lag, 1, self.input_hidden).to(self.device)
        self.astgcn = ASTGCN_submodule(self.device, nb_time_filter=self.temporal_hidden, in_channels=self.fea_dim + self.input_hidden)
        self.agcrn = AGCRN(params)

        self.gru = nn.GRU(input_size=self.input_hidden, hidden_size=self.temporal_hidden,
                                         num_layers=self.temporal_num_layer, batch_first=True)

        self.lstm = nn.LSTM(input_size=self.input_hidden + self.cat_embed_dim, hidden_size=self.temporal_hidden,
                                          num_layers=self.temporal_num_layer, batch_first=True)

        self.adaptive_t = TCN_block(in_fea=self.spatial_hidden, hidden=self.temporal_hidden,
                                            num_node=self.num_node, time_steps=10, device=self.device)

        # output
        self.output_layer = nn.Sequential(
            nn.Linear(160, self.output_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_hidden, 1)
        )
        self.mlp_output_layer = nn.Sequential(
            nn.Linear((self.input_hidden + self.cat_embed_dim) * self.time_lag, self.output_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_hidden, 1)
        )
        self.lstm_output_layer = nn.Sequential(
            nn.Linear(self.input_hidden, self.output_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_hidden, 1)
        )

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, mix_x, fea, A):

        day_embed = self.slice_embedding(fea[:, :, :, -1].long())
        conti_fea = self.input_layer(fea[:, :, :, :-1])
        conti_mix_fea = self.input_layer(mix_x[:, :, :, :-1])
        input_x = torch.cat([conti_fea, day_embed], dim=-1)
        input_x_mix = torch.cat([conti_mix_fea, day_embed], dim=-1)
        if self.model == 'GASTGCN':
            graph_s = self.adap_s(input_x)
            temporal_input = graph_s.permute(0, 2, 1, 3)
            graph_st = self.adaptive_t(temporal_input)
            graph_s_constra = self.adap_s(input_x_mix)
            temporal_input_constra = graph_s_constra.permute(0, 2, 1, 3)
            graph_st_constra = self.adaptive_t(temporal_input_constra)
            pred = self.output_layer(graph_st)
            return pred.squeeze(), graph_st_constra
        elif self.model == 'ISTGCN':
            pred = self.istgcn(torch.tensor(A).to(self.device).float(), fea.permute(0, 2, 1, 3).contiguous())
            return pred.squeeze(), fea
        elif self.model == 'LSTM':
            temporal_input = input_x.permute(0, 2, 1, 3)
            # temporal_input: [B, N, T, hidden] -> [B*N, T, hidden]
            temporal_input = temporal_input.reshape(-1, temporal_input.shape[2], temporal_input.shape[3])
            outputs, _ = self.lstm(temporal_input)
            outputs = outputs[:, -1, :]
            pred = self.lstm_output_layer(outputs).reshape(-1, self.num_node)
            return pred, fea
        elif self.model == 'MLP':
            pred = self.mlp_output_layer(input_x.permute(0, 2, 1, 3).reshape(input_x.shape[0], input_x.shape[2], -1))
            return pred.squeeze(), fea
        elif self.model == 'AGCRN':
            pred = self.agcrn(input_x).squeeze(1)
            return pred.squeeze(), fea
        elif self.model == 'ASTGCN':
            input_x = input_x.permute(0, 2, 3, 1)
            # graph_s: [B, N, 1]
            graph_st = self.astgcn(input_x)
            pred = self.astgcn_output(graph_st)

            return pred.squeeze(), fea
        else:
            print('please specify correct model name')


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


if __name__ == '__main__':
    pass
