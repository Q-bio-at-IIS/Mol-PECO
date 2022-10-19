#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-18 18:39:56
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DreamTestData(Dataset):
    """docstring for DreamTestData"""
    def __init__(self, data_dir, marker, wanted_dilution, wanted_label, norm):
        super(DreamTestData, self).__init__()
        self.wanted_label = wanted_label
        self.wanted_dilution = wanted_dilution
        if marker == "val":
            self.data_path = os.path.join(data_dir, "LBs2.txt")
            self.dilu_path = os.path.join(data_dir, "dilution_leaderboard.txt")
        else:
            self.data_path = os.path.join(data_dir, "GSs2_new.txt")
            self.dilu_path = os.path.join(data_dir, "dilution_testset.txt")
        self.cid2smiles_path = os.path.join(data_dir, "cid2smiles.txt")
        self.atom_dict_path = os.path.join(data_dir, "atom_dict.pkl")
        self.adj_dict_path = os.path.join(data_dir, "adj_dict.pkl")
        self.mol_mask_path = os.path.join(data_dir, "mol_masks.pkl")
        self.mol_size_path = os.path.join(data_dir, "mol_sizes.pkl")
        self.coulomb_dict_path = os.path.join(data_dir, "coulomb_dict.pkl")
        self.label_path = os.path.join(data_dir, "label.txt")

        self.eigVal_adj_path = os.path.join(data_dir, "eigVal_adj.pkl")
        self.eigVec_adj_path = os.path.join(data_dir, "eigVec_adj.pkl")
        self.eigVal_coulomb_path = os.path.join(data_dir, "eigVal_coulomb.pkl")
        self.eigVec_coulomb_path = os.path.join(data_dir, "eigVec_coulomb.pkl")

        assert norm in ["none", "zscore"]
        self.norm = norm

        self._read_cid2smiles()
        self._read_test_data()
        self._read_features()

    def set_mus_sigmas(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas

    def _read_test_data(self):
        test_df = pd.read_csv(self.data_path, sep = "\t", header = 0)
        test_ids = sorted(set(test_df["oID"].values.squeeze().tolist()))

        if self.wanted_dilution != "all":
            dilution = pd.read_csv(self.dilu_path, sep = "\t", header = 0)
            dilution["dilution"] = dilution["dilution"].str.strip()
            dilution["dilution"] = dilution["dilution"].str.replace(",", "")
            dilution["dilution"] = dilution["dilution"].str.replace("'", "")
            dilution["new_dilution"] = dilution["dilution"].str[2:]
            mask = dilution["new_dilution"] == self.wanted_dilution
            dilution = dilution[mask]
            dilution_ids = dilution["oID"].values.squeeze().tolist()
            oids = [test_id for test_id in test_ids if test_id in dilution_ids]
        else:
            oids = test_ids

        dfs = []
        for test_id in oids:
            mask = test_df["oID"] == test_id
            this_df = test_df[mask]
            this_df.pop("oID")
            this_df.pop("sigma")
            this_df = this_df.set_index("descriptor").T
            this_df["Compound Identifier"] = test_id
            dfs.append(this_df)
        self.test_df = pd.concat(dfs)
        print("{}: {}".format(self.wanted_dilution, len(self.test_df)))

    def _read_cid2smiles(self):
        cid2smiles = pd.read_csv(self.cid2smiles_path, sep = ",", header = None)
        cid2smiles = cid2smiles.rename(columns = {0:"CID", 1:"SMILES"})
        self.cid2smiles = cid2smiles.set_index('CID').T.to_dict('records')[0]

    def _read_features(self):
        self.atom_dict = pickle.load(open(self.atom_dict_path, "rb"))
        self.adj_dict = pickle.load(open(self.adj_dict_path, "rb"))
        self.mol_masks = pickle.load(open(self.mol_mask_path, "rb"))
        self.mol_sizes = pickle.load(open(self.mol_size_path, "rb"))
        self.coulomb_dict = pickle.load(open(self.coulomb_dict_path, "rb"))
        if self.wanted_label == "all":
            self.labels = pd.read_csv(self.label_path)["labels"].values.squeeze().tolist()
        else:
            self.labels = [self.wanted_label]

        self.eigval_adj = pickle.load(open(self.eigVal_adj_path, "rb"))
        self.eigvec_adj = pickle.load(open(self.eigVec_adj_path, "rb"))
        self.eigval_coulomb = pickle.load(open(self.eigVal_coulomb_path, "rb"))
        self.eigvec_coulomb = pickle.load(open(self.eigVec_coulomb_path, "rb"))

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        data = self.test_df.iloc[[idx]]
        cid = data["Compound Identifier"].values[0]
        res = [cid]
        for l in self.labels:
            v = data[l].values[0]
            res.append(v)
        return res

    def _adj2edge(self, adj):
        r_idxs, c_idxs = np.nonzero(adj)
        edges = np.array([r_idxs, c_idxs])
        return edges

    def _pro_label(self, data, start_idx, l):
        mini_data = data[:, start_idx:start_idx + 1]
        if self.norm == "none":
            mini_data = mini_data.astype(float)/100
            mini_data = mini_data.astype(float)
            nan_mask = np.isnan(mini_data)
            mini_data[nan_mask] = 0
            zero_mask = mini_data == 0
            weights = (~zero_mask).astype(int)
        elif self.norm == "zscore":
            mini_data = (mini_data.astype(float)-self.mus[l])/self.sigmas[l]
            mini_data = mini_data.astype(float)
            nan_mask = np.isnan(mini_data)
            mini_data[nan_mask] = 0
            zero_mask = mini_data == 0
            weights = (~zero_mask).astype(int)
        else:
            raise("normalization not implemented")           
        return mini_data, weights

    def collate_fn(self, data):
        data = np.array(data)
        data_dict = {}
        for i, l in enumerate(self.labels):
            data_dict[l], data_dict["{}_weight".format(l)] = self._pro_label(data, 1 + i, l)

        cids = data[:, 0]
        atom_arrays, adj_arrays, coulomb_arrays, mol_masks, mol_sizes, smiles_list = [], {}, {}, [], [], []
        eigval_adj, eigvec_adj, eigval_coulomb, eigvec_coulomb = [], [], [], []
        for cid in cids:
            smiles = self.cid2smiles[int(cid)]
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
                if k not in coulomb_arrays:
                    coulomb_arrays[k] = [v]
                else:
                    coulomb_arrays[k].append(v)
            smiles_list.append(smiles)
            eigval_adj.append(self.eigval_adj[smiles])
            eigvec_adj.append(self.eigvec_adj[smiles])
            eigval_coulomb.append(self.eigval_coulomb[smiles])
            eigvec_coulomb.append(self.eigvec_coulomb[smiles])
            
        data_dict["atom_feat"] = atom_arrays
        data_dict["adj_feat"] = adj_arrays
        data_dict["coulomb_feat"] = coulomb_arrays
        data_dict["mol_mask"] = mol_masks
        data_dict["mol_size"] = mol_sizes
        data_dict["smiles"] = smiles_list
        data_dict["eigval_adj"] = eigval_adj
        data_dict["eigvec_adj"] = eigvec_adj
        data_dict["eigval_coulomb"] = eigval_coulomb
        data_dict["eigvec_coulomb"] = eigvec_coulomb
        return data_dict

if __name__ == '__main__':
    data_dir = "../data_DREAM"
    train_data = DreamTestData(data_dir, "val")
    

