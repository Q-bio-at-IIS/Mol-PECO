#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-14 14:20:01
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
# @Link    : ${link}
# @Version : $Id$

import os, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConv, GraphConvSkip

def model_params(model):
    return np.sum([l.numel() for l in model.parameters()])/1e6

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class GNNEncoder(nn.Module):
    def __init__(self, atom_num, emb_dim, hid_dims, bonds = [0, 1, 2, 3]):
        super(GNNEncoder, self).__init__()
        print(bonds)
        self.embs = nn.Embedding(atom_num, emb_dim)
        self.bonds = bonds
        self.hid_dims = hid_dims

        self.gnn_layers_bond = {}
        for b in bonds:
            gnn_layers = self._gnn_layer_for_bond(emb_dim, hid_dims)
            self.gnn_layers_bond[b] = gnn_layers

        self._add_modules(self.gnn_layers_bond, "gcn")

    def _add_modules(self, layers, n):
        ##add modules
        modules = {}
        modules.update(layers)
        for k, v in modules.items():
            name = "{}_{}".format(n, k)
            self.add_module(name, v)

    def _gnn_layer_for_bond(self, emb_dim, hid_dims):
        gnn_layers = nn.ModuleList()
        in_dims = [emb_dim] + hid_dims[:-1]
        out_dims = hid_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            gnn_layer = GraphConvSkip(in_dim, out_dim)
            gnn_layers.append(gnn_layer)
        return gnn_layers

    def _forward_gnn_for_bond(self, bond_type, atom_embs, adjs):
        atom_embs_list = []
        for gnn in self.gnn_layers_bond[bond_type]:
            atom_embs = gnn(atom_embs, adjs)
            atom_embs_list.append(atom_embs)
        atom_embs = torch.cat(atom_embs_list, dim = -1)
        atom_embs = F.selu(atom_embs)
        return atom_embs

    def _synergistic_bonds(self, atom_embs, adjs):
        if len(adjs) > 1:
            cnter = 0
            for k, v, in adjs.items():
                if k not in self.bonds:
                    continue
                updated_emb = self._forward_gnn_for_bond(k, atom_embs, v)
                if cnter == 0:
                    sum_embs = updated_emb
                    mul_embs = updated_emb
                else:
                    sum_embs += updated_emb
                    mul_embs *= updated_emb
                cnter += 1
            mol_embs = sum_embs - mul_embs
        else:
            raise("Error in generating molecular embedding")
        return mol_embs

    def _summed_bonds(self, atom_embs, adjs):
        updated_embs = []
        for k, v, in adjs.items():
            if k not in self.bonds:
                continue
            updated_emb = self._forward_gnn_for_bond(k, atom_embs, v)
            updated_embs.append(updated_emb.unsqueeze(3))
        updated_embs = torch.cat(updated_embs, dim = -1)
        mol_embs = torch.sum(updated_embs, dim = -1)
        return mol_embs

    def forward(self, x, adjs, mask):
        atom_embs = self.embs(x)
        
        mol_embs = self._summed_bonds(atom_embs, adjs)
        # mol_embs = self._synergistic_bonds(atom_embs, adjs)

        loc_embs = mol_embs.view(-1, mol_embs.shape[-1])
        # loc_mask = mask.repeat(1, 1, sum(self.hid_dims))
        # loc_mask = loc_mask.view(-1, mol_embs.shape[-1])
        # loc_embs = loc_embs[loc_mask]

        mol_embs[~mask] = float('nan')
        glo_embs = torch.nansum(mol_embs, dim = 1)
        return glo_embs, loc_embs

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class GNN(nn.Module):
    """docstring for ClassName"""
    def __init__(self, atom_num, emb_dim, hid_dims, fc_dims, drop_rate, label_names, bonds = [0, 1, 2, 3]):
        super(GNN, self).__init__()
        self.label_names = label_names

        self.encoder = GNNEncoder(atom_num, emb_dim, hid_dims, bonds)

        self.fc_layers = nn.ModuleList()
        in_dims = [sum(hid_dims)] + fc_dims
        out_dims = fc_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            fc_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(drop_rate)
                )
            self.fc_layers.append(fc_layer)

        self.out_layers = {}
        for l in label_names:
            # out_layer = nn.Sequential(
            #     nn.Linear(fc_dims[-1], 1),
            #     nn.Sigmoid()
            #     )
            out_layer = nn.Linear(fc_dims[-1], 1)
            self.out_layers[l] = out_layer

        self._add_modules(self.out_layers, "fc")

    def _add_modules(self, layers, n):
        ##add modules
        modules = {}
        modules.update(layers)
        for k, v in modules.items():
            name = "{}_{}".format(n, k)
            self.add_module(name, v)

    def forward_emb(self, x, adjs, mask):
        mol_embs, _ = self.encoder(x, adjs, mask)
        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)
        return mol_embs        

    def forward(self, x, adjs, mask):
        mol_embs = self.forward_emb(x, adjs, mask)

        preds = {}
        for l in self.label_names:
            pred = self.out_layers[l](mol_embs)
            preds[l] = pred

        return preds

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class GNN1Out(nn.Module):
    """docstring for ClassName"""
    def __init__(self, atom_num, emb_dim, hid_dims, fc_dims, clf_dim, drop_rate, bonds = [0, 1, 2, 3]):
        super(GNN1Out, self).__init__()

        self.encoder = GNNEncoder(atom_num, emb_dim, hid_dims, bonds)

        self.fc_layers = nn.ModuleList()
        in_dims = [sum(hid_dims)] + fc_dims
        out_dims = fc_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            fc_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(drop_rate)
                )
            self.fc_layers.append(fc_layer)

        self.out_layers = nn.Sequential(
            nn.Linear(fc_dims[-1], clf_dim),
            nn.Softmax(dim = -1)
            )

    def forward_emb(self, x, adjs, mask):
        mol_embs, _ = self.encoder(x, adjs, mask)
        # print("**"*20)
        # print(mol_embs)
        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)
            # print(mol_embs)
        # print("=="*20)
        return mol_embs        

    def forward(self, x, adjs, mask):
        mol_embs = self.forward_emb(x, adjs, mask)

        preds = self.out_layers(mol_embs)

        return preds

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class SimpleNN(nn.Module):
    """docstring for ClassName"""
    def __init__(self, atom_num, emb_dim, hid_dims, fc_dims, drop_rate, label_names, bonds = [0, 1, 2, 3]):
        super(SimpleNN, self).__init__()
        self.label_names = label_names

        self.embs = nn.Embedding(atom_num, emb_dim)


        self.fc_layers = nn.ModuleList()
        in_dims = [emb_dim] + hid_dims + fc_dims
        out_dims = hid_dims + fc_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            fc_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Tanh(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(drop_rate)
                )
            self.fc_layers.append(fc_layer)

        self.out_layers = {}
        for l in label_names:
            out_layer = nn.Linear(fc_dims[-1], 1)
            self.out_layers[l] = out_layer

        self._add_modules(self.out_layers, "fc")

    def _add_modules(self, layers, n):
        ##add modules
        modules = {}
        modules.update(layers)
        for k, v in modules.items():
            name = "{}_{}".format(n, k)
            self.add_module(name, v)


    def forward(self, x, adjs):##x [batch, atoms], edges [2, edge_num]
        atom_embs = self.embs(x)
        mol_embs = torch.mean(atom_embs, dim = 1)

        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)

        preds = {}
        for l in self.label_names:
            pred = self.out_layers[l](mol_embs)
            preds[l] = pred

        return preds

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

if __name__ == '__main__':
    model = GNN(10, 32, [64, 96], [96, 64], 21, 0.1)
    model.apply(weight_init)
    print("model params: {}".format(model_params(model)))
    for k, v in model.named_parameters():
        print(k, v.size())