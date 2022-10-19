#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-02 20:19:11
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
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

class GNNEncoderAdjCou(nn.Module):
    def __init__(self, atom_num, emb_dim, hid_dims, bonds_adj = [0, 1, 2, 3], bonds_cou = ["coulomb"]):
        super(GNNEncoderAdjCou, self).__init__()
        print(bonds_adj, bonds_cou)
        self.embs = nn.Embedding(atom_num, emb_dim)
        self.bonds_adj = bonds_adj
        self.bonds_cou = bonds_cou
        self.hid_dims = hid_dims

        self.gnn_layers_bond = {}
        for b in bonds_adj:
            gnn_layers = self._gnn_layer_for_bond(emb_dim, hid_dims)
            self.gnn_layers_bond[b] = gnn_layers

        self.gnn_layers_cou = {}
        for b in bonds_cou:
            gnn_layers = self._gnn_layer_for_bond(emb_dim, hid_dims)
            self.gnn_layers_cou[b] = gnn_layers

        self._add_modules(self.gnn_layers_bond, "gcn_bond")
        self._add_modules(self.gnn_layers_cou, "gcn_coulomb")

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

    def _forward_gnn_for_bond(self, gnn_layers, bond_type, atom_embs, adjs):
        atom_embs_list = []
        for gnn in gnn_layers[bond_type]:
            atom_embs = gnn(atom_embs, adjs)
            atom_embs_list.append(atom_embs)
        atom_embs = torch.cat(atom_embs_list, dim = -1)
        atom_embs = F.selu(atom_embs)
        return atom_embs

    def _summed_bonds(self, atom_embs, adjs, adj = True):
        if adj:
            bonds = self.bonds_adj
            gnn_layers = self.gnn_layers_bond
        else:
            bonds = self.bonds_cou
            gnn_layers = self.gnn_layers_cou

        updated_embs = []
        for k, v, in adjs.items():
            if k not in bonds:
                continue
            updated_emb = self._forward_gnn_for_bond(gnn_layers, k, atom_embs, v)
            updated_embs.append(updated_emb.unsqueeze(3))
        updated_embs = torch.cat(updated_embs, dim = -1)
        mol_embs = torch.sum(updated_embs, dim = -1)
        return mol_embs

    def _gen_mol_embs(self, adj, cou):
        return adj + cou - adj*cou

    def forward(self, x, adjs, coulomb, mask):
        atom_embs = self.embs(x)
        
        adj_embs = self._summed_bonds(atom_embs, adjs)
        cou_embs = self._summed_bonds(atom_embs, coulomb, False)
        mol_embs = self._gen_mol_embs(adj_embs, cou_embs)

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

class GNNAdjCou(nn.Module):
    """docstring for ClassName"""
    def __init__(self, atom_num, emb_dim, hid_dims, fc_dims, drop_rate, label_names, bonds_adj = [0, 1, 2, 3], bonds_cou = ["coulomb"]):
        super(GNNAdjCou, self).__init__()
        self.label_names = label_names

        self.encoder = GNNEncoderAdjCou(atom_num, emb_dim, hid_dims, bonds_adj, bonds_cou)

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

    def forward_emb(self, x, adjs, coulomb, mask):
        mol_embs, _ = self.encoder(x, adjs, coulomb, mask)
        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)
        return mol_embs        

    def forward(self, x, adjs, coulomb, mask):
        mol_embs = self.forward_emb(x, adjs, coulomb, mask)

        preds = {}
        for l in self.label_names:
            pred = self.out_layers[l](mol_embs)
            preds[l] = pred

        return preds

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()