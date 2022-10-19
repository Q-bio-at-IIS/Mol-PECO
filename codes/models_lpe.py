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

class LearnedPE(nn.Module):

    def __init__(self, LPE_dim = 8, LPE_n_heads = 4, LPE_layer = 2, max_freq = 5):
        super(LearnedPE, self).__init__()
        self.linear_A = nn.Linear(2, LPE_dim)        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layer)
        self.max_freq = max_freq

    def forward(self, EigVecs_b, EigVals_b): ## EigVecs_b [B, N, m]
        PosEnc_b = []
        for EigVecs, EigVals in zip(EigVecs_b, EigVals_b):
            EigVecs, EigVals = EigVecs[:, :self.max_freq], EigVals[:, :self.max_freq]
            PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals.unsqueeze(2)), dim=2).float() # [N, m, 2]
            empty_mask = torch.isnan(PosEnc) # [N, m, 2]

            PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
            PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
            PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
            
            #1st Transformer: Learned PE
            PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
            
            #remove masked sequences
            PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
            
            #Sum pooling
            PosEnc = torch.nansum(PosEnc, 0, keepdim=True)
            PosEnc_b.append(PosEnc)

        PosEnc_b = torch.cat(PosEnc_b, dim = 0)
        return PosEnc_b

class GNNEncoder(nn.Module):
    def __init__(
        self, atom_num, emb_dim, hid_dims, 
        LPE_dim = 8, LPE_n_heads = 4, LPE_layer = 2, max_freq = 5, 
        bonds = [0, 1, 2, 3]
        ):
        super(GNNEncoder, self).__init__()
        print(bonds)
        self.embs_lpe = LearnedPE(LPE_dim, LPE_n_heads, LPE_layer, max_freq)
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

    def forward(self, atoms, adjs, EigVecs, EigVals, mask):
        atom_embs_lpe = self.embs_lpe(EigVecs, EigVals)
        atom_embs_raw = self.embs(atoms)
        atom_embs = atom_embs_lpe + atom_embs_raw - atom_embs_raw*atom_embs_lpe

        mol_embs_lpe, _ = self._forward_atom_embs(atom_embs_lpe, adjs, mask)
        mol_embs_raw, _ = self._forward_atom_embs(atom_embs_raw, adjs, mask)
        mol_embs, loc_embs = self._forward_atom_embs(atom_embs, adjs, mask)

        return mol_embs_raw, mol_embs_lpe, mol_embs, loc_embs
        
    def _forward_atom_embs(self, atom_embs, adjs, mask):
        mol_embs = self._summed_bonds(atom_embs, adjs)
        loc_embs = mol_embs.view(-1, mol_embs.shape[-1])

        mol_embs[~mask] = float('nan')
        glo_embs = torch.nansum(mol_embs, dim = 1)
        return glo_embs, loc_embs

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class GNNLPE(nn.Module):
    """docstring for ClassName"""
    def __init__(
        self, atom_num, emb_dim, hid_dims, fc_dims, drop_rate, label_names, 
        LPE_dim = 8, LPE_n_heads = 4, LPE_layer = 2, max_freq = 5, 
        bonds = [0, 1, 2, 3]):
        super(GNNLPE, self).__init__()
        self.label_names = label_names

        self.encoder = GNNEncoder(atom_num, emb_dim, hid_dims, LPE_dim, LPE_n_heads, LPE_layer, max_freq, bonds)

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

    def forward_emb(self, atoms, adjs, EigVecs, EigVals, mask):
        mol_embs_raw, mol_embs_lpe, mol_embs, _ = self.encoder(atoms, adjs, EigVecs, EigVals, mask)
        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)
            mol_embs_raw = fc(mol_embs_raw)
            mol_embs_lpe = fc(mol_embs_lpe)
        return mol_embs_raw, mol_embs_lpe, mol_embs        

    def forward(self, atoms, adjs, EigVecs, EigVals, mask):
        mol_embs_raw, mol_embs_lpe, mol_embs = self.forward_emb(atoms, adjs, EigVecs, EigVals, mask)

        preds, preds_raw, preds_lpe = {}, {}, {}
        for l in self.label_names:
            pred = self.out_layers[l](mol_embs)
            preds[l] = pred

            pred_raw = self.out_layers[l](mol_embs_raw)
            preds_raw[l] = pred_raw

            pred_lpe = self.out_layers[l](mol_embs_lpe)
            preds_lpe[l] = pred_lpe
        return preds, preds_raw, preds_lpe

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
