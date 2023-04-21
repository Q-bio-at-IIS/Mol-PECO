#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 18:38:21
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$
import os, pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PyrfumeData(Dataset):
    """docstring for PyrfumeData"""
    def __init__(self, data_dir, data_name, masked_coulomb = False):
        super(PyrfumeData, self).__init__()
        self.data_path = os.path.join(data_dir, data_name)
        self.atom_dict_path = os.path.join(data_dir, "atom_dict.pkl")
        self.adj_dict_path = os.path.join(data_dir, "adj_dict.pkl")
        self.mol_mask_path = os.path.join(data_dir, "mol_masks.pkl")
        self.mol_size_path = os.path.join(data_dir, "mol_sizes.pkl")
        self.label_path = os.path.join(data_dir, "labels.xlsx")
        self.data_name = data_name

        self.coulomb_dict_path = os.path.join(data_dir, "coulomb_dict.pkl")

        self.eigVal_adj_path = os.path.join(data_dir, "eigVal_adj.pkl")
        self.eigVec_adj_path = os.path.join(data_dir, "eigVec_adj.pkl")
        self.eigVal_coulomb_path = os.path.join(data_dir, "eigVal_coulomb.pkl")
        self.eigVec_coulomb_path = os.path.join(data_dir, "eigVec_coulomb.pkl")

        self._read_features()
        self._read_train_data()

        self.masked_coulomb = masked_coulomb

    def _read_train_data(self):
        data_df = pd.read_excel(self.data_path, header = 0)
        data_df["SMILES"] = data_df["SMILES"].str.strip()
        smiles_list = data_df["SMILES"].values.squeeze()
        mask = np.array([True if s in self.tot_smiles else False for s in smiles_list])
        self.data_df = data_df[mask]
        # print("#samples of {}: {}".format(self.data_name, len(self.data_df)))

        ## cnt the number of samples for different labels
        tot_num = self.data_df.shape[0]
        tot_labels = []
        for idx in range(tot_num):
            data = self.data_df.iloc[[idx]]
            labels = self._extract_odor(data["Odor"].values[0])
            tot_labels.append(labels)

        self.label_weights = {}
        for l in self.labels:
            cnt = 0
            for labels in tot_labels:
                if l in labels:
                    cnt += 1
            ratio = cnt/tot_num
            weight = 1 - ratio
            self.label_weights[l] = weight
            # print(l, self.label_weights[l])
        print("**"*20)

    def _read_features(self):
        self.atom_dict = pickle.load(open(self.atom_dict_path, "rb"))
        self.adj_dict = pickle.load(open(self.adj_dict_path, "rb"))
        self.mol_masks = pickle.load(open(self.mol_mask_path, "rb"))
        self.mol_sizes = pickle.load(open(self.mol_size_path, "rb"))
        self.labels = pd.read_excel(self.label_path)["labels"].values.squeeze().tolist()
        self.tot_smiles = list(self.atom_dict.keys())

        self.coulomb_dict = pickle.load(open(self.coulomb_dict_path, "rb"))

        self.eigval_adj = pickle.load(open(self.eigVal_adj_path, "rb"))
        self.eigvec_adj = pickle.load(open(self.eigVec_adj_path, "rb"))
        self.eigval_coulomb = pickle.load(open(self.eigVal_coulomb_path, "rb"))
        self.eigvec_coulomb = pickle.load(open(self.eigVec_coulomb_path, "rb"))

    def _extract_odor(self, description):
        labels = [t.strip() for t in description.strip().replace("[", "").replace("]", "").replace("'", "").split(",")]
        return labels

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data = self.data_df.iloc[[idx]]
        smiles = data["SMILES"].values[0]
        labels = self._extract_odor(data["Odor"].values[0])
        outs = [smiles]
        for l in self.labels:
            if l in labels:
                outs.append(1)
            else:
                outs.append(0)
        return outs

    def _pro_label(self, data, start_idx):
        mini_data = data[:, start_idx:start_idx + 1]
        mini_data = mini_data.astype(float)
        nan_mask = np.isnan(mini_data)
        mini_data[nan_mask] = 0

        zero_mask = mini_data == 0
        weights = np.zeros_like(mini_data)
        weights[zero_mask] = (1 - np.sum(zero_mask)/len(zero_mask))
        weights[~zero_mask] = np.sum(zero_mask)/len(zero_mask)
        return mini_data, weights

    def collate_fn(self, data):
        data = np.array(data)
        data_dict = {}
        for i, l in enumerate(self.labels):
            data_dict[l], data_dict["{}_weight".format(l)] = self._pro_label(data, i+1)

        smiles_list = data[:, 0]
        atom_arrays, adj_arrays, coulomb_arrays, mol_masks, mol_sizes = [], {}, {}, [], []
        eigval_adj, eigvec_adj, eigval_coulomb, eigvec_coulomb = [], [], [], []
        for smiles in smiles_list:
            smiles = smiles.strip()
            atom_array, adj_array, coulomb_array, mol_mask, mol_size = self.atom_dict[smiles], self.adj_dict[smiles], self.coulomb_dict[smiles], self.mol_masks[smiles], self.mol_sizes[smiles]
            atom_arrays.append(atom_array)

            mol_masks.append(mol_mask)
            mol_sizes.append(mol_size)
            for i, adj in enumerate(adj_array):
                if i not in adj_arrays:
                    adj_arrays[i] = [adj]
                else:
                    adj_arrays[i].append(adj)

            for k,v in coulomb_array.items():
                if self.masked_coulomb:
                    mask = np.sum(adj_array, axis = 0).astype(bool)
                    v[~mask] = 0

                if k not in coulomb_arrays:
                    coulomb_arrays[k] = [v]
                else:
                    coulomb_arrays[k].append(v)

            eigval_adj.append(self.eigval_adj[smiles])
            eigvec_adj.append(self.eigvec_adj[smiles])
            eigval_coulomb.append(self.eigval_coulomb[smiles])
            eigvec_coulomb.append(self.eigvec_coulomb[smiles])
        
        data_dict["smiles"] = smiles_list
        data_dict["atom_feat"] = atom_arrays
        data_dict["adj_feat"] = adj_arrays
        data_dict["mol_mask"] = mol_masks
        data_dict["mol_size"] = mol_sizes
        data_dict["label_weights"] = self.label_weights
        data_dict["coulomb_feat"] = coulomb_arrays
        data_dict["eigval_adj"] = eigval_adj
        data_dict["eigvec_adj"] = eigvec_adj
        data_dict["eigval_coulomb"] = eigval_coulomb
        data_dict["eigvec_coulomb"] = eigvec_coulomb
        return data_dict

if __name__ == '__main__':

    data_dir = "../pyrfume_models3_sois/dumped30_coulomb_zscore/"
    train_data = PyrfumeData(data_dir, "pyrfume_train.xlsx", masked_coulomb = True)
    for d in train_data:
        print(d)
        break
