#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-03 13:33:58
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$
import os, torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class TesterCoulomb(object):
    """docstring for Tester"""
    def __init__(self, model, test_loader, labels, out_dir, method = "zscore"):
        super(TesterCoulomb, self).__init__()
        self.model = model.cuda()
        self.criterion = nn.BCELoss()
        self.test_loader = test_loader
        self.labels_list = labels
        self.out_dir = out_dir
        self.method = method

        self.model.eval()

    def _to_var(self, x, t = "float"):
        x = np.array(x)
        if t == "int":
            return torch.LongTensor(x).cuda()
        elif t == "bool":
            return torch.BoolTensor(x).cuda()
        else:
            return torch.FloatTensor(x).cuda()

    def _to_var_dict(self, x):
        new_dict = {}
        for k, v in x.items():
            new_dict[k] = self._to_var(v, t = "float")
        return new_dict

    def _to_np(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().squeeze()
        else:
            return np.array(x).squeeze()

    def _cal_pearson(self, preds, labels):
        rs = []
        for l in self.labels_list:
            r, p = pearsonr(self._to_np(preds[l]), self._to_np(labels[l]))
            rs.append(r)
        return rs

    def _data_to_var(self, data):
        atom_feats, coulomb_feats = data["atom_feat"], data["coulomb_feat"]
        eigval_coulomb, eigvec_coulomb = data["eigval_coulomb"], data["eigvec_coulomb"]
        mol_masks = data["mol_mask"]
        atom_feats = self._to_var(atom_feats, t = "int")
        coulomb_feats = self._to_var_dict(coulomb_feats)
        eigval_coulomb = self._to_var(eigval_coulomb, t = "float")
        eigvec_coulomb = self._to_var(eigvec_coulomb, t = "float")
        mol_masks = self._to_var(mol_masks, t = "bool")
        return atom_feats, coulomb_feats, eigval_coulomb, eigvec_coulomb, mol_masks

    def test_on_step(self, data):
        atom_feats, cou_feats, eigval_coulomb, eigvec_coulomb, mol_masks = self._data_to_var(data)
        preds = self.model(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)
        return preds

    def generate_emb_on_step(self, data):
        atom_feats, cou_feats, eigval_coulomb, eigvec_coulomb, mol_masks = self._data_to_var(data)
        mol_embs_raw, mol_embs_lpe, mol_embs = self.model.forward_emb(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)
        return mol_embs

    def _update_total_dict(self, total_dict, dicts):
        for l in self.labels_list:
            if l in total_dict:
                total_dict[l] = total_dict[l] + self._to_np(dicts[l]).tolist()
            else:
                total_dict[l] = self._to_np(dicts[l]).squeeze().tolist()
        return total_dict

    def _scatter_plot(self, total_preds, total_labels, rs, out_name):
        out_dir = os.path.join(self.out_dir, out_name)
        os.makedirs(out_dir, exist_ok = True)
        for i, l in enumerate(self.labels_list):
            preds = np.array(total_preds[l])
            labels = np.array(total_labels[l])
            fig = plt.figure(figsize = (3,3), dpi = 300)
            plt.scatter(labels, preds, alpha = 0.8, color = "red")
            m = LinearRegression(fit_intercept = False)
            m.fit(labels.reshape(-1, 1), preds.reshape(-1, 1))
            xs = np.arange(-0.05, 1.10, 0.05).reshape(-1, 1)
            pred_b = m.predict(xs)
            plt.plot(xs, pred_b, color = "blue", lw = 2, alpha = 0.8)
            plt.xlabel("True")
            plt.ylabel("Pred")
            plt.title("Pearson of {}".format(round(rs[i], 3)))
            # plt.xlim(-0.05, 1.05)
            # plt.ylim(-0.05, 1.05) 
            plt.savefig(os.path.join(out_dir, "{}.png".format(l)), bbox_inches = "tight")
        out_df = pd.DataFrame()
        out_df["label"] = self.labels_list
        out_df["Pearson"] = rs
        out_df.to_excel(os.path.join(out_dir, "pearson.xlsx"), index = False)

    def _roc(self, total_preds, total_labels, out_name):
        out_dir = os.path.join(self.out_dir, out_name)
        fig = plt.figure(figsize = (10,10), dpi = 300)
        os.makedirs(out_dir, exist_ok = True)
        aucs = []
        for l in self.labels_list:
            preds = np.array(total_preds[l])
            labels = np.array(total_labels[l])
            fpr, tpr, ths = metrics.roc_curve(labels, preds)
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label = "{}: {}".format(l, round(auc, 3)))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("1 - Specificity")
            plt.ylabel("Sensitivity")
            aucs.append(auc)
        plt.legend()
        plt.title("mean auc of {}".format(round(np.nanmean(aucs), 3)))
        plt.savefig(os.path.join(out_dir, "roc.png"), bbox_inches = "tight")
        return np.nanmean(aucs)

    def predict_regression(self, out_name):
        # self._log_param()
        total_preds, total_trues = {}, {}
        for data in self.test_loader:
            preds = self.test_on_step(data)
            total_preds = self._update_total_dict(total_preds, preds)
            total_trues = self._update_total_dict(total_trues, data)
        rs = self._cal_pearson(total_preds, total_trues)
        self._scatter_plot(total_preds, total_trues, rs, out_name)
        return rs

    def predict_classification(self, out_name):
        total_preds, total_trues = {}, {}
        for data in self.test_loader:
            preds = self.test_on_step(data)
            total_preds = self._update_total_dict(total_preds, preds)
            total_trues = self._update_total_dict(total_trues, data)
        auc = self._roc(total_preds, total_trues, out_name)
        return auc     

    def _log_param(self):
        for k, v in self.model.named_parameters():
            print(k, v)

    def generate_emb(self, marker):
        # self._log_param()
        tot_mol_embs, labels, label_weights, smiles_list = [], {}, {}, []
        for data in self.test_loader:
            mol_embs = self.generate_emb_on_step(data)
            mol_embs = mol_embs.detach().cpu().numpy()
            tot_mol_embs.append(mol_embs)
            smiles_list.extend(list(data["smiles"]))
            for l in self.labels_list:
                ls = data[l]
                ls_weight = data["{}_weight".format(l)]
                if l in labels:
                    labels[l].append(ls)
                    label_weights[l].append(ls_weight)
                else:
                    labels[l] = [ls]
                    label_weights[l] = [ls_weight]
        tot_mol_embs = np.vstack(tot_mol_embs)
        out_df = pd.DataFrame(data = tot_mol_embs, columns = ["emb{}".format(i) for i in range(mol_embs.shape[1])])
        for l in self.labels_list:
            labels[l] = np.vstack(labels[l])
            label_weights[l] = np.vstack(label_weights[l])
            out_df[l] = labels[l]
            out_df["{}_weight".format(l)] = label_weights[l]
        out_df["split"] = [marker]*len(out_df)
        out_df["smiles"] = smiles_list
        return out_df
