#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-15 14:17:26
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$
import os, pickle, torch
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch.nn.functional as F
from tqdm import tqdm
from scipy import sparse as sp
from chem_utils import laplace_decomp, cal_coulomb_matrix

class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self, cou_norm = "frobenius", add_Hs = False, kekulize = True, max_atoms = -1, max_size = -1, self_loop = False, step_binary = 10):
        super(Preprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.max_atoms = max_atoms
        self.max_size = max_size
        self.self_loop = self_loop
        self.cou_norm = cou_norm
        self.step_binary = step_binary
        print("normlize coulomb matrix with {}".format(cou_norm))

    def _prepare_mol(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol)
            if self.add_Hs:
                mol = Chem.AddHs(mol)
            if self.kekulize:
                Chem.Kekulize(mol)
            return mol, canonical_smiles, True
        except:
            return None, None, False

    def _check_num_atoms(self, mol, num_max_atom = -1):
        num_atoms = mol.GetNumAtoms()
        if num_max_atom >= 0 and num_atoms > num_max_atom:
            return False
        else:
            return True

    def _construct_atomic_number_array(self, mol, max_size = -1):
        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        n_atom = len(atom_list)
        if max_size < 0:
            return np.array(atom_list, dtype = np.int32)
        else:
            atom_array = np.zeros(max_size, dtype = np.int32)
            atom_array[:n_atom] = np.array(atom_list, dtype = np.int32)
            return atom_array

    def _construct_discrete_edge_matrix(self, mol, max_size = -1):
        N = mol.GetNumAtoms()
        if max_size < 0:
            size = N
        else:
            size = max_size
        adjs = np.zeros((4, size, size), dtype = np.float32)
        bond2channel = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3
        }
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            ch = bond2channel[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjs[ch, i, j] = 1.
            adjs[ch, j, i] = 1.
            if self.self_loop:
                adjs[ch, i, i] = 1.
                adjs[ch, j, j] = 1.
        return adjs

    def _sigmoid(self, arr):
        return 1./(1+np.exp(-arr))

    def _binarilize(self, arr):
        bin_arr = arr > 0.5
        bin_arr = bin_arr.astype("int")
        return bin_arr

    def _norm_coulomb_mat(self, data, mol_size, epsilon = 1e-9): ## normalization in matrix-wise
        # print(data[:mol_size, :mol_size])
        ## perform normalization
        if self.cou_norm == "minmax":
            min_ = np.min(data, keepdims = True)
            max_ = np.max(data, keepdims = True)
            new_data = {"coulomb": (data - min_)/(max_ - min_ + epsilon)}
        elif self.cou_norm == "frobenius":
            norm = np.linalg.norm(data, keepdims = True)
            # print(norm)
            new_data = {"coulomb": data/(norm + epsilon)}
            # print(new_data["coulomb"][:mol_size,:mol_size])
        elif self.cou_norm == "none":
            new_data = {"coulomb": data}
        else:
            raise("normalization of coulomb is {}, which is not implemented".format(self.method))
        if ~self.check_symmetric(new_data["coulomb"][:mol_size,:mol_size]):
            print(new_data["coulomb"][:mol_size,:mol_size])
            print("\n\n\n")
        # print(new_data["coulomb"][:mol_size,:mol_size])
        # print("=="*20)
        return new_data

    def _norm_coulomb_row(self, data, mol_size, epsilon = 1e-9): ## z-score
        # print(data[:mol_size, :mol_size])
        if self.cou_norm == "zscore":
            mu = np.mean(data, axis = -1, keepdims = True)
            sigma = np.std(data, axis = -1, keepdims = True)
            new_data = {"coulomb": (data-mu)/(sigma + epsilon)}
        elif self.cou_norm == "minmax":
            min_ = np.min(data, axis = -1, keepdims = True)
            max_ = np.max(data, axis = -1, keepdims = True)
            new_data = {"coulomb": (data - min_)/(max_ - min_ + epsilon)}
        elif self.cou_norm == "frobenius":
            norm = np.linalg.norm(data, axis = -1, keepdims = True)
            new_data = {"coulomb": data/(norm + epsilon)}
        elif self.cou_norm == "binary":
            new_data = {"coulomb": data}
            new_data["coulomb0"] = self._binarilize(self._sigmoid(data))
            for s in range(1, self.step_binary):
                new_data["coulomb{}".format(s)] = self._binarilize(self._sigmoid(data + s))
                new_data["coulomb{}".format(-s)] = self._binarilize(self._sigmoid(data - s))
        elif self.cou_norm == "none":
            new_data = {"coulomb": data}
        else:
            raise("normalization of coulomb is {}, which is not implemented".format(self.method))
        # print(new_data["coulomb"][:mol_size,:mol_size])
        # print("=="*20)
        return new_data["coulomb"]

    def check_symmetric(self, a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)

    def _get_features(self, mol):
        if not self._check_num_atoms(mol, self.max_atoms):
            return None, None, None, mol.GetNumAtoms(), None, False

        atom_array = self._construct_atomic_number_array(mol, self.max_size)
        adj_array = self._construct_discrete_edge_matrix(mol, self.max_size)
        coulomb_array, status = cal_coulomb_matrix(mol, self.max_size)
        mol_size = mol.GetNumAtoms()
        weighted_am, coulomb_lpe = None, None
        if status:
            weighted_am = self._norm_coulomb_mat(coulomb_array, mol_size)
            coulomb_lpe = self._norm_coulomb_row(coulomb_array, mol_size)
        
        return atom_array, adj_array, weighted_am, mol_size, coulomb_lpe, status

    def process(self, smiles):
        mol, canonical_smiles, status = self._prepare_mol(smiles)
        if status:
            atom_array, adj_array, weighted_am, mol_size, coulomb_lpe, flag = self._get_features(mol)
            return atom_array, adj_array, weighted_am, mol_size, canonical_smiles, coulomb_lpe, flag
        else:
            return None, None, None, None, None, None, False

class DataDumperLoader(object):
    """docstring for ClassName"""
    def __init__(self, data_dir, cou_norm = "frobenius", max_atoms = 28, max_size = 28, atom2id_path = "", laplace = False, freq = 10, add_Hs = False, self_loop = False, step_binary = 10):
        super(DataDumperLoader, self).__init__()
        os.makedirs(data_dir, exist_ok = True)
        self.max_atoms = max_atoms
        self.max_size = max_size
        self.data_dir = data_dir
        self.processor = Preprocessor(cou_norm = cou_norm, max_atoms = max_atoms, max_size = max_size, add_Hs = add_Hs, self_loop = self_loop, step_binary = step_binary)
        self.atom2id = self._load_data_dict(atom2id_path)
        self.laplace = laplace
        self.freq = freq
        self.read_smiles_list()

    def read_smiles_list(self):
        pass

    def _build_mol_mask(self, mol_size):
        if self.max_atoms != -1:
            mask = np.zeros((self.max_atoms))
        else:
            mask = np.zeros((mol_size))
        mask[:mol_size] = 1
        mask = mask.astype(bool)
        return mask

    def _cal_laplace_adj(self, adj):
        A = np.sum(adj, axis = 0).clip(0, 1)
        EigVals, EigVecs = laplace_decomp(A, max_freqs = self.freq)
        return EigVals.numpy(), EigVecs.numpy()

    def _cal_laplace_coulomb(self, coulomb, mol_size):
        # print(coulomb[:mol_size, :mol_size])
        EigVals, EigVecs = laplace_decomp(coulomb, max_freqs = self.freq)
        return EigVals.numpy(), EigVecs.numpy()

    def _process(self):
        self.atom_dict, self.adj_dict, self.coulomb_dict, self.mol_masks, self.mol_sizes = {}, {}, {}, {}, {}
        self.eigVal_adj, self.eigVec_adj, self.eigVal_coulomb, self.eigVec_coulomb = {}, {}, {}, {}
        for smiles in tqdm(self.smiles_list, ncols = 80):
            atom_array, adj_array, weighted_am, mol_size, canonical_smiles, coulomb_lpe, flag = self.processor.process(smiles)

            if not flag:
                continue
            atoms, status = self._build_atom2id(atom_array)
            if status:
                self.atom_dict[smiles] = atoms
                self.adj_dict[smiles] = adj_array 
                self.mol_masks[smiles] = self._build_mol_mask(mol_size)
                self.mol_sizes[smiles] = mol_size
                self.coulomb_dict[smiles] = weighted_am
                self.eigVal_adj[smiles], self.eigVec_adj[smiles] = self._cal_laplace_adj(adj_array)
                self.eigVal_coulomb[smiles], self.eigVec_coulomb[smiles] = self._cal_laplace_coulomb(coulomb_lpe, mol_size)
            else:
                print("{} not in atom2id".format(smiles))
            # break
            
    def _process_example(self, smiles = "C1CCN(CC1)C(=O)C=CC=CC2=CC3=C(C=C2)OCO3"):
        atom_array, adj_array, coulomb_array, mol_size, canonical_smiles, _, flag = self.processor.process(smiles)
        eigVal_adj, eigVec_adj = self._cal_laplace_adj(adj_array)
        eigVal_coulomb, eigVec_coulomb = self._cal_laplace_coulomb(coulomb_array["coulomb"], 10)
        return atom_array, adj_array, coulomb_array, mol_size, canonical_smiles, eigVal_adj, eigVec_adj, eigVal_coulomb, eigVec_coulomb

    def _build_atom2id(self, atom_array):
        ids, status = [], True
        for atom in atom_array:
            if atom in self.atom2id:
                ids.append(self.atom2id[atom])
            else:
                status = False
                return ids, status
        return np.array(ids), status

    def _dump_data_dict(self, data, data_name):
        with open(os.path.join(self.data_dir, data_name), "w+") as f:
            for k, v in data.items():
                f.write("{}\t{}\n".format(k, v))
        f.close()

    def _load_data_dict(self, data_path):
        df = pd.read_csv(data_path, sep = "\t", header = None)
        df.columns = ["atom", "id"]
        data_dict = dict(zip(df['atom'], df['id']))
        return data_dict

    def dump(self):
        self._process()
        self._dump_data_dict(self.atom2id, "atom2id.txt")
        pickle.dump(self.mol_masks, open(os.path.join(self.data_dir, "mol_masks.pkl"), "wb"))
        pickle.dump(self.coulomb_dict, open(os.path.join(self.data_dir, "coulomb_dict.pkl"), "wb"))
        pickle.dump(self.atom_dict, open(os.path.join(self.data_dir, "atom_dict.pkl"), "wb"))       
        pickle.dump(self.adj_dict, open(os.path.join(self.data_dir, "adj_dict.pkl"), "wb"))
        pickle.dump(self.mol_sizes, open(os.path.join(self.data_dir, "mol_sizes.pkl"), "wb"))
        pickle.dump(self.eigVal_adj, open(os.path.join(self.data_dir, "eigVal_adj.pkl"), "wb"))
        pickle.dump(self.eigVec_adj, open(os.path.join(self.data_dir, "eigVec_adj.pkl"), "wb"))
        pickle.dump(self.eigVal_coulomb, open(os.path.join(self.data_dir, "eigVal_coulomb.pkl"), "wb"))
        pickle.dump(self.eigVec_coulomb, open(os.path.join(self.data_dir, "eigVec_coulomb.pkl"), "wb"))

    def load(self):
        atom_dict = pickle.load(open(os.path.join(self.data_dir, "atom_dict.pkl"), "rb"))
        adj_dict = pickle.load(open(os.path.join(self.data_dir, "adj_dict.pkl"), "rb"))
        coulomb_dict = pickle.load(open(os.path.join(self.data_dir, "coulomb_dict.pkl"), "rb"))
        return atom_dict, adj_dict, coulomb_dict

class PyrfumeDumper(DataDumperLoader):
    def __init__(self, data_dir, cou_norm = "frobenius", max_atoms = 28, max_size = 28, atom2id_path = "", laplace = False, freq = 10, add_Hs = False, self_loop = False, step_binary = 10):
        super(PyrfumeDumper, self).__init__(data_dir, cou_norm, max_atoms, max_size, atom2id_path, laplace, freq, add_Hs, self_loop, step_binary)

    def read_smiles_list(self):
        smiles_list = pd.read_csv(os.path.join(self.data_dir, "cid2smiles.txt"), header = None)[1].values.squeeze().tolist()
        smiles_set = set(smiles_list)
        self.smiles_list = sorted(list(smiles_set))

        train_path = os.path.join(self.data_dir, "pyrfume_train.xlsx")
        val_path = os.path.join(self.data_dir, "pyrfume_val.xlsx")
        test_path = os.path.join(self.data_dir, "pyrfume_test.xlsx")
        train_df = pd.read_excel(train_path)
        val_df = pd.read_excel(val_path)
        test_df = pd.read_excel(test_path)

        train_smiles = train_df["SMILES"].values.squeeze().tolist()
        val_smiles = val_df["SMILES"].values.squeeze().tolist()
        test_smiles = test_df["SMILES"].values.squeeze().tolist()
        # for t in train_smiles + val_smiles + test_smiles:
        #     print(t)
        #     print(t.strip())
        #     print("=="*20) 
        self.smiles_list = self.smiles_list + sorted([t.strip() for t in train_smiles + val_smiles + test_smiles])
        print("smiles: {}".format(len(self.smiles_list)))

if __name__ == '__main__':
    data_dir = "../data/CM_frobenius/"
    atom2id_path = "../data/atom2id.txt"
    dumper = PyrfumeDumper(data_dir, step_binary = 20, cou_norm = "minmax", max_atoms = 62, max_size = 62, atom2id_path = atom2id_path, add_Hs = True, freq = 20)
    dumper.dump()
