#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-15 14:17:26
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$
import os, pickle
import numpy as np
import pandas as pd
import rdkit.Chem as Chem

class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self, add_Hs = False, kekulize = True, max_atoms = -1, max_size = -1):
        super(Preprocessor, self).__init__()
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.max_atoms = max_atoms
        self.max_size = max_size

    def _prepare_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        if self.add_Hs:
            model = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return mol, canonical_smiles

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
        return adjs

    def _get_features(self, mol):
        if not self._check_num_atoms(mol, self.max_atoms):
            return None, None, mol.GetNumAtoms(), False

        atom_array = self._construct_atomic_number_array(mol, self.max_size)
        adj_array = self._construct_discrete_edge_matrix(mol, self.max_size)
        mol_size = mol.GetNumAtoms()
        return atom_array, adj_array, mol_size, True

    def process(self, smiles):
        mol, canonical_smiles = self._prepare_mol(smiles)
        atom_array, adj_array, mol_size, flag = self._get_features(mol)
        return atom_array, adj_array, mol_size, canonical_smiles, flag

class DataDumperLoader(object):
    """docstring for ClassName"""
    def __init__(self, data_dir, max_atoms = 28, max_size = 28):
        super(DataDumperLoader, self).__init__()
        self.max_atoms = max_atoms
        self.max_size = max_size
        self.data_dir = data_dir
        self.processor = Preprocessor(max_atoms = max_atoms, max_size = max_size)
        self.atom2id = {}

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

    def _process(self):
        self.atom_dict, self.adj_dict, self.mol_masks, self.mol_sizes = {}, {}, {}, {}
        for smiles in self.smiles_list:
            atom_array, adj_array, mol_size, canonical_smiles, flag = self.processor.process(smiles)
            if not flag:
                print(smiles, mol_size)
                continue
            self.atom_dict[smiles] = self._build_atom2id(atom_array)
            self.adj_dict[smiles] = adj_array 
            self.mol_masks[smiles] = self._build_mol_mask(mol_size)
            self.mol_sizes[smiles] = mol_size

    def _build_atom2id(self, atom_array):
        ids = []
        for atom in atom_array:
            if atom not in self.atom2id:
                self.atom2id[atom] = len(self.atom2id)
            ids.append(self.atom2id[atom])
        return np.array(ids)

    def _dump_data_dict(self, data, data_name):
        with open(os.path.join(self.data_dir, data_name), "w+") as f:
            for k, v in data.items():
                f.write("{}\t{}\n".format(k, v))
        f.close()

    def dump(self):
        self.read_smiles_list()
        self._process()
        self._dump_data_dict(self.atom2id, "atom2id.txt")
        pickle.dump(self.mol_masks, open(os.path.join(self.data_dir, "mol_masks.pkl"), "wb"))
        pickle.dump(self.atom_dict, open(os.path.join(self.data_dir, "atom_dict.pkl"), "wb"))       
        pickle.dump(self.adj_dict, open(os.path.join(self.data_dir, "adj_dict.pkl"), "wb"))
        pickle.dump(self.mol_sizes, open(os.path.join(self.data_dir, "mol_sizes.pkl"), "wb"))

    def load(self):
        atom_dict = pickle.load(open(os.path.join(self.data_dir, "atom_dict.pkl"), "rb"))
        adj_dict = pickle.load(open(os.path.join(self.data_dir, "adj_dict.pkl"), "rb"))
        return atom_dict, adj_dict

class DREAMDumper(DataDumperLoader):
    def __init__(self, data_dir, max_atoms = 28, max_size = 28):
        super(DREAMDumper, self).__init__(data_dir, max_atoms, max_size)

    def read_smiles_list(self):
        smiles_list = pd.read_csv(os.path.join(self.data_dir, "cid2smiles.txt"), header = None)[1].values.squeeze().tolist()
        smiles_set = set(smiles_list)
        self.smiles_list = sorted(list(smiles_set))

class AICROWDDumper(DataDumperLoader):
    def __init__(self, data_dir, max_atoms = 28, max_size = 28):
        super(AICROWDDumper, self).__init__(data_dir, max_atoms, max_size)

    def read_smiles_list(self):
        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "round-3-supplementary-training-data.csv")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_smiles, test_smiles = train_df["SMILES"].values.squeeze().tolist(), test_df["SMILES"].values.squeeze().tolist()
        self.smiles_list = sorted([t.strip() for t in train_smiles + test_smiles])
        print("smiles: {}".format(len(self.smiles_list)))

class DREAMAICROWDDumper(DataDumperLoader):
    def __init__(self, data_dir, max_atoms = 28, max_size = 28):
        super(DREAMAICROWDDumper, self).__init__(data_dir, max_atoms, max_size)

    def read_smiles_list(self):
        smiles_list = pd.read_csv(os.path.join(self.data_dir, "cid2smiles.txt"), header = None)[1].values.squeeze().tolist()
        smiles_set = set(smiles_list)
        self.smiles_list = sorted(list(smiles_set))

        train_path = os.path.join(self.data_dir, "train.csv")
        test_path = os.path.join(self.data_dir, "round-3-supplementary-training-data.csv")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_smiles, test_smiles = train_df["SMILES"].values.squeeze().tolist(), test_df["SMILES"].values.squeeze().tolist()
        self.smiles_list = self.smiles_list + sorted([t.strip() for t in train_smiles + test_smiles])
        print("smiles: {}".format(len(self.smiles_list)))

if __name__ == '__main__':
    # data_dir = "../data_DREAM/"
    # dumper = DREAMDumper(data_dir, max_atoms = 28, max_size = 28)
    # dumper.dump()
    # atom_dict, adj_dict = dumper.load()
    # atom_sets = set()
    # for k, v in atom_dict.items():
    #     print(k, v)
    #     break     

    # data_dir = "../data_AIcrowd/"
    # dumper = AICROWDDumper(data_dir, max_atoms = 28, max_size = 28)
    # dumper.dump()
    # atom_dict, adj_dict = dumper.load()
    # mol_sizes = [len(t) for t in atom_dict.values()]
    # print(max(mol_sizes), np.percentile(mol_sizes, 95))
    # atom_sets = set()
    # for k, v in atom_dict.items():
    #     print(k, v)
    #     break    

    data_dir = "../dumped_dream_aicrowd/"
    dumper = DREAMAICROWDDumper(data_dir, max_atoms = 28, max_size = 28)
    dumper.dump()
    atom_dict, adj_dict = dumper.load()
    mol_sizes = [len(t) for t in atom_dict.values()]
    print(max(mol_sizes), np.percentile(mol_sizes, 95))
    atom_sets = set()
    for k, v in atom_dict.items():
        print(k, v)
        break    