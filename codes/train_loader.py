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

class DreamTrainData(Dataset):
    """docstring for DreamTrainData"""
    def __init__(self, data_dir, wanted_dilution, method = "mean", wanted_label = "all", norm = "none"):
        super(DreamTrainData, self).__init__()
        self.wanted_dilution = wanted_dilution
        self.method = method
        self.wanted_label = wanted_label
        self.data_path = os.path.join(data_dir, "TrainSet.txt")
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
        self._read_train_data()
        self._read_features()
        self._cal_norm()

    def _read_train_data(self):
        train_df = pd.read_csv(self.data_path, sep = "\t", header = 0)
        if self.wanted_dilution != "all":
            train_df["Dilution"] = train_df["Dilution"].str.strip()
            train_df["Dilution"] = train_df["Dilution"].str.replace(",", "")
            train_df["Dilution"] = train_df["Dilution"].str.replace("'", "")
            train_df["Dilution_new"] = train_df["Dilution"].str[2:]
            mask = train_df["Dilution_new"] == self.wanted_dilution
            train_df = train_df[mask]
        else:
            train_df = train_df
        if self.method == "mean":
            self.train_df = train_df.groupby("Compound Identifier").mean().reset_index()
        else:
            self.train_df = train_df
        print("{}: {}".format(self.wanted_dilution, len(self.train_df)))

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

    def _cal_norm(self):
        self.mus, self.sigmas = {}, {}
        for l in self.labels:
            values = self.train_df[l].values.squeeze()
            mu = np.mean(values)
            sigma = np.std(values)
            self.mus[l] = mu
            self.sigmas[l] = sigma
            print(l, mu, sigma)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        data = self.train_df.iloc[[idx]]
        cid = data["Compound Identifier"].values[0]
        dilution = self.wanted_dilution
        subject = data["subject #"].values[0]
        res = [cid, dilution, subject]
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
            data_dict[l], data_dict["{}_weight".format(l)] = self._pro_label(data, 3 + i, l)

        cids = data[:, 0]
        atom_arrays, adj_arrays, coulomb_arrays, mol_masks, mol_sizes, smiles_list = [], {}, {}, [], [], []
        eigval_adj, eigvec_adj, eigval_coulomb, eigvec_coulomb = [], [], [], []
        for cid in cids:
            smiles = self.cid2smiles[int(cid)]
            if smiles not in self.atom_dict:
                print(smiles, " not in preprocessed smiles list")
                continue
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

class AICrowdData(Dataset):
    """docstring for AICrowdData"""
    def __init__(self, data_dir, data_name):
        super(AICrowdData, self).__init__()
        self.data_path = os.path.join(data_dir, data_name)
        self.atom_dict_path = os.path.join(data_dir, "atom_dict.pkl")
        self.adj_dict_path = os.path.join(data_dir, "adj_dict.pkl")
        self.mol_mask_path = os.path.join(data_dir, "mol_masks.pkl")
        self.mol_size_path = os.path.join(data_dir, "mol_sizes.pkl")
        self.label_path = os.path.join(data_dir, "tot_vocabulary.csv")
        self.data_name = data_name

        self._read_features()
        self._read_train_data()

    def _read_train_data(self):
        data_df = pd.read_csv(self.data_path, header = 0)
        data_df["SMILES"] = data_df["SMILES"].str.strip()
        smiles_list = data_df["SMILES"].values.squeeze()
        mask = np.array([True if s in self.tot_smiles else False for s in smiles_list])
        self.data_df = data_df[mask]
        print("#samples of {}: {}".format(self.data_name, len(self.data_df)))

    def _read_features(self):
        self.atom_dict = pickle.load(open(self.atom_dict_path, "rb"))
        self.adj_dict = pickle.load(open(self.adj_dict_path, "rb"))
        self.mol_masks = pickle.load(open(self.mol_mask_path, "rb"))
        self.mol_sizes = pickle.load(open(self.mol_size_path, "rb"))
        self.labels = pd.read_csv(self.label_path)["label"].values.squeeze().tolist()
        self.tot_smiles = list(self.atom_dict.keys())

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data = self.data_df.iloc[[idx]]
        smiles = data["SMILES"].values[0]
        labels = data["SENTENCE"].values[0].strip().split(",")

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
        atom_arrays, adj_arrays, mol_masks, mol_sizes = [], {}, [], []
        for smiles in smiles_list:
            smiles = smiles.strip()
            atom_array, adj_array, mol_mask, mol_size = self.atom_dict[smiles], self.adj_dict[smiles], self.mol_masks[smiles], self.mol_sizes[smiles]
            atom_arrays.append(atom_array)
            mol_masks.append(mol_mask)
            mol_sizes.append(mol_size)
            for i, adj in enumerate(adj_array):
                if i not in adj_arrays:
                    adj_arrays[i] = [adj]
                else:
                    adj_arrays[i].append(adj)
        data_dict["atom_feat"] = atom_arrays
        data_dict["adj_feat"] = adj_arrays
        data_dict["mol_mask"] = mol_masks
        data_dict["mol_size"] = mol_sizes

        return data_dict

class GoodScentsData(Dataset):
    """docstring for AICrowdData"""
    def __init__(self, data_dir, data_name):
        super(GoodScentsData, self).__init__()
        self.data_path = os.path.join(data_dir, data_name)
        self.atom_dict_path = os.path.join(data_dir, "atom_dict.pkl")
        self.adj_dict_path = os.path.join(data_dir, "adj_dict.pkl")
        self.mol_mask_path = os.path.join(data_dir, "mol_masks.pkl")
        self.mol_size_path = os.path.join(data_dir, "mol_sizes.pkl")
        self.coulomb_dict_path = os.path.join(data_dir, "coulomb_dict.pkl")
        self.label_path = os.path.join(data_dir, "labels.xlsx")
        self.data_name = data_name

        self.eigVal_adj_path = os.path.join(data_dir, "eigVal_adj.pkl")
        self.eigVec_adj_path = os.path.join(data_dir, "eigVec_adj.pkl")
        self.eigVal_coulomb_path = os.path.join(data_dir, "eigVal_coulomb.pkl")
        self.eigVec_coulomb_path = os.path.join(data_dir, "eigVec_coulomb.pkl")

        self._read_features()
        self._read_train_data()

    def _read_train_data(self):
        data_df = pd.read_excel(self.data_path, header = 0)
        data_df["smiles"] = data_df["smiles"].str.strip()
        smiles_list = data_df["smiles"].values.squeeze()
        mask = np.array([True if s in self.tot_smiles else False for s in smiles_list])
        self.data_df = data_df[mask]
        print("#samples of {}: {}".format(self.data_name, len(self.data_df)))

    def _read_features(self):
        self.atom_dict = pickle.load(open(self.atom_dict_path, "rb"))
        self.adj_dict = pickle.load(open(self.adj_dict_path, "rb"))
        self.mol_masks = pickle.load(open(self.mol_mask_path, "rb"))
        self.mol_sizes = pickle.load(open(self.mol_size_path, "rb"))
        self.coulomb_dict = pickle.load(open(self.coulomb_dict_path, "rb"))
        self.labels = pd.read_excel(self.label_path)["labels"].values.squeeze().tolist()
        self.tot_smiles = list(self.atom_dict.keys())

        self.eigval_adj = pickle.load(open(self.eigVal_adj_path, "rb"))
        self.eigvec_adj = pickle.load(open(self.eigVec_adj_path, "rb"))
        self.eigval_coulomb = pickle.load(open(self.eigVal_coulomb_path, "rb"))
        self.eigvec_coulomb = pickle.load(open(self.eigVec_coulomb_path, "rb"))

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data = self.data_df.iloc[[idx]]
        smiles = data["smiles"].values[0]
        labels = data["odor"].values[0].strip().replace("['", "").replace("']", "").split(",")

        outs = [smiles]
        for l in self.labels:
            if l in labels:
                outs.append(1)
            else:
                outs.append(0)
        return outs

    def collate_fn(self, data):
        data = np.array(data)
        data_dict = {}

        smiles_list = data[:, 0]
        atom_arrays, adj_arrays, coulomb_arrays, mol_masks, mol_sizes = [], {}, [], [], []
        eigval_adj, eigvec_adj, eigval_coulomb, eigvec_coulomb = [], [], [], []
        for smiles in smiles_list:
            smiles = smiles.strip()
            atom_array, adj_array, coulomb_array, mol_mask, mol_size = self.atom_dict[smiles], self.adj_dict[smiles], self.coulomb_dict[smiles], self.mol_masks[smiles], self.mol_sizes[smiles]
            atom_arrays.append(atom_array)
            coulomb_arrays.append(coulomb_array)
            mol_masks.append(mol_mask)
            mol_sizes.append(mol_size)
            for i, adj in enumerate(adj_array):
                if i not in adj_arrays:
                    adj_arrays[i] = [adj]
                else:
                    adj_arrays[i].append(adj)

            eigval_adj.append(self.eigval_adj[smiles])
            eigvec_adj.append(self.eigvec_adj[smiles])
            eigval_coulomb.append(self.eigval_coulomb[smiles])
            eigvec_coulomb.append(self.eigvec_coulomb[smiles])
        data_dict["smiles"] = smiles_list
        data_dict["atom_feat"] = atom_arrays
        data_dict["adj_feat"] = adj_arrays
        data_dict["coulomb_feat"] = coulomb_arrays
        data_dict["mol_mask"] = mol_masks
        data_dict["mol_size"] = mol_sizes
        data_dict["labels"] = np.argmax(data[:, 1: ], axis = -1)

        data_dict["eigval_adj"] = eigval_adj
        data_dict["eigvec_adj"] = eigvec_adj
        data_dict["eigval_coulomb"] = eigval_coulomb
        data_dict["eigvec_coulomb"] = eigvec_coulomb
        return data_dict

class PyrfumeData(Dataset):
    """docstring for PyrfumeData"""
    def __init__(self, data_dir, data_name):
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

    def _read_train_data(self):
        data_df = pd.read_excel(self.data_path, header = 0)
        data_df["SMILES"] = data_df["SMILES"].str.strip()
        smiles_list = data_df["SMILES"].values.squeeze()
        mask = np.array([True if s in self.tot_smiles else False for s in smiles_list])
        self.data_df = data_df[mask]
        print("#samples of {}: {}".format(self.data_name, len(self.data_df)))

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
                if k not in coulomb_arrays:
                    coulomb_arrays[k] = [v]
                else:
                    coulomb_arrays[k].append(v)

            eigval_adj.append(self.eigval_adj[smiles])
            eigvec_adj.append(self.eigvec_adj[smiles])
            eigval_coulomb.append(self.eigval_coulomb[smiles])
            eigvec_coulomb.append(self.eigvec_coulomb[smiles])
            
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
    # data_dir = "../data_DREAM"
    # train_data = DreamTrainData(data_dir, "100000", "individual", "intensity")
    
    
    # data_dir = "../data_AIcrowd"
    # train_data = AICrowdData(data_dir, data_name)
    # for d in train_data:
    #     print(d)
    #     break


    # data_dir = "../dumped_dream_goodscents_assign_atom2id"
    # train_data = GoodScentsData(data_dir, "goodscents_train.xlsx")
    # for d in train_data:
    #     print(d)
    #     break

    data_dir = "../dream_pyrfume_models/dumped_coulomb_minmax/"
    train_data = PyrfumeData(data_dir, "GdLef_train.xlsx")
    for d in train_data:
        print(d)
        break
