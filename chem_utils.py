#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-09-12 16:14:56
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import deepchem as dc
import numpy as np
import torch.nn.functional as F
import rdkit.Chem as Chem
from scipy import sparse as sp

def cal_coulomb_matrix(mol, max_atoms = 28):
    try:
        featurizer = dc.feat.CoulombMatrix(max_atoms = max_atoms)
        cou_matrix = featurizer.coulomb_matrix(mol)
        if np.isinf(cou_matrix).sum()!=0:
            print(Chem.MolToSmiles(mol), "invalid coulomb matrix")
            return None, False
        return cou_matrix[0], True
    except Exception as e:
        return None, False

def laplace_decomp(A, max_freqs = 10):
    # Laplacian
    n = A.shape[0]
    N = sp.diags(A.sum(axis = 0).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L)
    # print(EigVals.shape, EigVecs.shape)
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)

    EigVals = EigVals.repeat(n, 1)
    return EigVals, EigVecs

if __name__ == '__main__':
    smiles = "CCCCOC(=O)CC(CC(=O)OCCCC)(C(=O)OCCCC)OC(=O)C"
    mol = Chem.MolFromSmiles(smiles)
    print(mol.GetNumAtoms())
    mol = Chem.AddHs(mol)
    print(mol.GetNumAtoms())
    cou_matrix = cal_coulomb_matrix(mol, max_atoms = mol.GetNumAtoms())
    print(cou_matrix.shape)
    # print(cou_matrix[0])

    # N = mol.GetNumAtoms()
    # adjs = np.zeros((N, N), dtype = np.float32)
    # bond2channel = {
    #         Chem.BondType.SINGLE: 0,
    #         Chem.BondType.DOUBLE: 1,
    #         Chem.BondType.TRIPLE: 2,
    #         Chem.BondType.AROMATIC: 3
    # }
    # for bond in mol.GetBonds():
    #     i = bond.GetBeginAtomIdx()
    #     j = bond.GetEndAtomIdx()
    #     adjs[i, j] = 1.
    #     adjs[j, i] = 1.
    # print(adjs.shape)
    # print(adjs)
