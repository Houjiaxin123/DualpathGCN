from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer
from models.model_utils import *
from models.H2GCNmodel import *

class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class PatchGCN_Surv(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg=['spatial', None], multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=4):
        super(PatchGCN_Surv, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(1536, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU()])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)    

    def forward(self,  **kwargs):
        data = kwargs['x_path']
                
        if self.edge_agg[0] == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg[0] == 'latent':
            edge_index = data.edge_latent

        batch = data.batch
        edge_attr = None

        x = self.fc(data.x)
        x_ = x 
        
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return {"hazards": hazards, "S": S, "Y_hat": Y_hat, "logits": logits, "h": h}


class CombinedModel(nn.Module):
    def __init__(self, patch_gcn_surv_model, h2gcn_model, edge_agg=['spatial', 'spatial'], n_classes=4, hidden_dim=512, fusion_dropout=0.25):
        super(CombinedModel, self).__init__()
        self.edge_agg = edge_agg
        self.patch_gcn_surv = patch_gcn_surv_model
        self.h2gcn = h2gcn_model

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )

        self.classifier = nn.Linear(hidden_dim, n_classes)


    def forward(self, **kwargs):
        data = kwargs['x_path']

        out1 = self.patch_gcn_surv(x_path=data)
        hazards1 = out1["hazards"]
        S1 = out1["S"]
        Y_hat1 = out1["Y_hat"]
        logits1 = out1["logits"]
        h1 = out1["h"]

        out2 = self.h2gcn(x_path=data)
        hazards2 = out2["hazards"]
        S2 = out2["S"]
        Y_hat2 = out2["Y_hat"]
        logits2 = out2["logits"]
        h2 = out2["h"]

        h1 = self.norm1(h1)
        h2 = self.norm2(h2)

        h_cat = torch.cat([h1, h2], dim=0)
        g = self.gate(h_cat)

        # gated fusion
        h_fused = g * h1 + (1.0 - g) * h2
        h_fused = self.fusion_proj(h_fused)

        logits = self.classifier(h_fused).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)


        return {"hazards": hazards, "S": S, "Y_hat": Y_hat, "logits": logits, "h": h_fused}
