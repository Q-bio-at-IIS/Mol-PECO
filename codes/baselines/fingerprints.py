#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-18 10:45:11
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from mordred import Calculator, descriptors
from rdkit.Chem import rdFingerprintGenerator

def cal_mordred(smiles_list):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    filtered_smiles, mols = [], []
    for smiles in tqdm(smiles_list, ncols = 80):
        mol = Chem.MolFromSmiles(smiles)
        if mol != None:
            mols.append(mol)
            filtered_smiles.append(smiles)
        else:
            print("Error in {}".format(smiles))

    fps = calc.pandas(mols).values
    smiles2fp = {smiles: fp for smiles, fp in zip(filtered_smiles, fps)}
    return smiles2fp

def cal_morgan(smiles_list, category = "count"):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius = 3, fpSize = 2048, countSimulation = False)
    smiles2fp = {}
    for smiles in tqdm(smiles_list, ncols = 80):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if category == "count":
                fp = fpgen.GetCountFingerprintAsNumPy(mol)
            elif category == "bit":
                fp = fpgen.GetFingerprintAsNumPy(mol)
            else:
                raise("Not implemented of {}".format(category))
            smiles2fp[smiles] = fp
        except Exception as e:
            print("Error in {}".format(smiles))
    return smiles2fp

class FPDumperLoader(object):
    """docstring for ClassName"""
    def __init__(self, data_dir):
        super(FPDumperLoader, self).__init__()
        self.data_dir = data_dir
        self.read_smiles_list()

    def read_smiles_list(self):
        train = os.path.join(self.data_dir, "pyrfume_train.xlsx")
        val = os.path.join(self.data_dir, "pyrfume_val.xlsx")
        test = os.path.join(self.data_dir, "pyrfume_test.xlsx")

        train_smiles = pd.read_excel(train)["SMILES"].values.squeeze().tolist()
        val_smiles = pd.read_excel(val)["SMILES"].values.squeeze().tolist()
        test_smiles = pd.read_excel(test)["SMILES"].values.squeeze().tolist()
        self.smiles_list = train_smiles + val_smiles + test_smiles
        print("total smiles of {}".format(len(self.smiles_list)))

    def _process(self):
        self.mordreds = cal_mordred(self.smiles_list)
        self.cfps = cal_morgan(self.smiles_list, "count")
        self.bfps = cal_morgan(self.smiles_list, "bit")

    def dump(self):
        self._process()
        pickle.dump(self.mordreds, open(os.path.join(self.data_dir, "mordreds.pkl"), "wb"))
        pickle.dump(self.cfps, open(os.path.join(self.data_dir, "cfps.pkl"), "wb"))
        pickle.dump(self.bfps, open(os.path.join(self.data_dir, "bfps.pkl"), "wb"))       

    def load(self):
        mordreds = pickle.load(open(os.path.join(self.data_dir, "mordreds.pkl"), "rb"))
        cfps = pickle.load(open(os.path.join(self.data_dir, "cfps.pkl"), "rb"))
        bfps = pickle.load(open(os.path.join(self.data_dir, "bfps.pkl"), "rb"))
        return mordreds, cfps, bfps


if __name__ == '__main__':
    data_dir = "../../pyrfume_sois_canon/dumped30_coulomb_frobenius/"
    dumper = FPDumperLoader(data_dir)
    dumper.dump()

    # smiles_list = ["CCCCCCCCC=O", "CCCC=O", "CCCCCC1CCC(=O)O1"]
    # cal_mordred(smiles_list)