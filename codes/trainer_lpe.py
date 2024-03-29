#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-10 19:19:44
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from matplotlib import pyplot as plt

import matplotlib.pylab as pylab
font = {"axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16}
pylab.rcParams.update(font)


class TrainerCoulomb(object):
    """docstring for TrainerCoulomb"""
    def __init__(self, model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr, lambdas):
        super(TrainerCoulomb, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epoch = epoch
        self.labels_list = labels
        self.out_dir = out_dir
        self.lambdas = lambdas
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = lr)
        lpe_params = list(map(id, self.model.encoder.embs_lpe.parameters()))
        base_params = filter(lambda p: id(p) not in lpe_params, self.model.parameters())
        self.optimizer = optim.Adam([
            {"params": self.model.encoder.embs_lpe.parameters(), lr:lr},
            {"params": base_params, lr:1.}
            ])


        self.scheduler = OneCycleLR(self.optimizer, max_lr = lr/10., steps_per_epoch = len(self.train_loader), epochs = self.epoch)
        # self.criterion = nn.L1Loss(reduction = "none")
        self.criterion = nn.BCELoss(reduction = "none")
        self.writer = SummaryWriter(os.path.join(out_dir, "logging"))
        os.makedirs(self.out_dir, exist_ok = True)

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
            return x.squeeze()

    def _call_loss_log(self, preds, labels, epsilon = 1e-9):
        losses = 0 
        for l in self.labels_list:
            weight = self._to_var(labels["{}_weight".format(l)].squeeze())
            pred, label = preds[l], self._to_var(labels[l])
            # print(l, torch.max(pred), torch.min(pred), pred)
            pred = torch.sigmoid(pred)
            mae = torch.mean(self.criterion(pred, label)*weight)
            log = torch.abs(torch.log(pred + epsilon) - torch.log(label + epsilon))
            mean_log = torch.mean(log*weight)
            this_loss = mae + mean_log
            this_loss *= labels["label_weights"][l]

            losses += this_loss

        return losses/len(self.labels_list)

    def draw_preds(self, preds, labels, mark, thresholds):
        
        out_dir = os.path.join(self.out_dir, mark)
        os.makedirs(out_dir, exist_ok = True)
        for l in self.labels_list:
            if l not in thresholds:
                continue
            xs = np.arange(len(preds[l]))
            df = pd.DataFrame()
            df["no"] = xs
            df["pred"] = preds[l] >= thresholds[l]
            df["label"] = labels[l]
            fig = plt.figure()
            sns.scatterplot(data = df, x = "no", y = "pred", hue = "label")
            plt.savefig(os.path.join(out_dir, "{}.png".format(l)), bbox_inches = "tight")

    def _cal_auc_auprc_metrics(self, preds, labels):
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, pred_neg_nums, thresholds = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for l in tqdm(self.labels_list, ncols = 80):
            if sum(labels[l]) == 0:
                continue
            ## dump the preds
            out_dir = os.path.join(self.out_dir, "detailed_preds")
            os.makedirs(out_dir, exist_ok = True)
            out_df = pd.DataFrame()
            out_df["label"] = labels[l]
            out_df["pred"] = preds[l]
            out_df.to_csv(os.path.join(out_dir, "{}.csv".format(l)))

            auc = metrics.roc_auc_score(labels[l], preds[l])
            auprc = metrics.average_precision_score(labels[l], preds[l])

            threshold = self._cal_youden(preds[l], labels[l])
            precision, recall, specificity, f1, acc = self._cal_metrics((preds[l] >= threshold).astype(int), labels[l])

            aucs[l] = auc
            nums[l] = sum(labels[l])/len(labels[l])
            pred_neg_nums[l] = sum(~(preds[l] >= threshold))/len(labels[l])
            auprcs[l] = auprc
            precisions[l] = precision
            recalls[l] = recall
            specificities[l] = specificity
            f1s[l] = f1
            accs[l] = acc
            thresholds[l] = threshold

        return aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, pred_neg_nums, thresholds

    def _cal_youden(self, preds, labels):
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        idx = np.argmax(tpr - fpr)
        return thresholds[idx]

    def _cal_max_acc(self, preds, labels):
        thresholds, accuracies = [], []
        for p in np.unique(preds):
            thresholds.append(p)
            preds_int = (preds >= p).astype(int)
            accuracies.append(metrics.balanced_accuracy_score(labels, preds_int))
        argmax = np.argmax(accuracies)
        return thresholds[argmax]

    def _cal_metrics(self, preds, labels):
        precision = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        specificity = metrics.recall_score(labels, preds, pos_label = 0)
        f1 = metrics.f1_score(labels, preds)
        acc = metrics.balanced_accuracy_score(labels, preds)
        return precision, recall, specificity, f1, acc

    def _data_to_var(self, data):
        atom_feats, cou_feats, mol_masks = data["atom_feat"], data["coulomb_feat"], data["mol_mask"]
        eigval_coulomb, eigvec_coulomb = data["eigval_coulomb"], data["eigvec_coulomb"]

        atom_feats = self._to_var(atom_feats, t = "int")
        cou_feats = self._to_var_dict(cou_feats)
        mol_masks = self._to_var(mol_masks, t = "bool")
        eigval_coulomb = self._to_var(eigval_coulomb, t = "float")
        eigvec_coulomb = self._to_var(eigvec_coulomb, t = "float")
        return atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb

    def _check_input(self, data):
        atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb = data
        # print(torch.sum(torch.isnan(atom_feats)))
        # print(torch.sum(torch.isnan(cou_feats["coulomb"])))
        # print(torch.sum(torch.isnan(mol_masks)))
        # print(torch.sum(torch.isnan(eigval_coulomb)))
        # print(torch.sum(torch.isnan(eigvec_coulomb)))

    def train_on_step(self, data):
        atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb = self._data_to_var(data)
        self._check_input((atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb))
        self.optimizer.zero_grad()
        # print(atom_feats.shape)
        preds = self.model(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)
        total_loss = self._call_loss_log(preds, data)
        # total_loss.register_hook(lambda grad: print(grad))
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # for n, p in self.model.named_parameters():
        #     try:
        #         gradient, *_ = p.grad.data
        #         print(f"Gradient of {n} w.r.t to L: {gradient}")
        #     except Exception as e:
        #         pass
        # print("\n")

        return total_loss.item(), preds

    def val_on_step(self, data):
        atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb = self._data_to_var(data)
        preds = self.model(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)
        total_loss = self._call_loss_log(preds, data)
        return total_loss.item(), preds

    def save_model(self, epoch, name):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "{}.ckpt".format(name)))

    def _update_total_dict(self, total_dict, dicts):
        for l in self.labels_list:
            if l in total_dict:
                try:
                    total_dict[l] = total_dict[l] + self._to_np(dicts[l]).tolist()
                except Exception as e:
                    total_dict[l].append(self._to_np(dicts[l]))
            else:
                total_dict[l] = self._to_np(dicts[l]).tolist()
        return total_dict

    def fit_classification(self):
        best_loss, best_auc = 9999, -1
        e_auc, e_loss = 0, 0
        pbar = tqdm(range(self.epoch), ncols = 120)
        for e in pbar:
            train_loss_tot, val_data, val_preds, val_loss_tot, val_auc = 0, {}, {}, 0, []

            self.model.train()
            for data in self.train_loader:
                if len(data["atom_feat"]) == 1:
                    continue
                total_loss, preds = self.train_on_step(data)
                train_loss_tot += total_loss
            train_loss_tot /= len(self.train_loader)

            self.model.eval()
            for data in self.val_loader:
                total_loss, val_pred = self.val_on_step(data)
                val_loss_tot += total_loss
                val_preds = self._update_total_dict(val_preds, val_pred)
                val_data = self._update_total_dict(val_data, data)
            val_loss_tot /= len(self.val_loader)


            val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, _, _ = self._cal_auc_auprc_metrics(val_preds, val_data)

            self.writer.add_scalars("Loss_tot", {"train": train_loss_tot, "validation": val_loss_tot}, e)
            self.writer.add_scalars("AUC (validation)", val_aucs, e)
            self.writer.add_scalars("AUPRC (validation)", val_auprcs, e)
            self.writer.add_scalars("Precision (validation)", val_precisions, e)
            self.writer.add_scalars("Recall (validation)", val_recalls, e)
            self.writer.add_scalars("Specificity (validation)", val_specificities, e)
            self.writer.add_scalars("F1 (validation)", val_f1s, e)
            self.writer.add_scalars("Acc (validation)", val_accs, e)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], e)
            if val_loss_tot < best_loss:
                self.save_model(e, "min_loss")
                best_loss = val_loss_tot
                e_loss = e
                pbar.set_description("min loss at {}: {}\tmax auc at {}: {}".format(e_loss, round(best_loss, 2), e_auc, round(best_auc, 2)))

            if np.mean(list(val_aucs.values())) > best_auc:
                best_auc = np.mean(list(val_aucs.values()))
                e_auc = e
                self.save_model(e, "max_auc")
                pbar.set_description("min loss at {}: {}\tmax auc at {}: {}".format(e_loss, round(best_loss, 2), e_auc, round(best_auc, 2)))

    def _draw_performances(self, nums, metrics, name, mark):
        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            if l in metrics:
                ax.scatter(nums[l], metrics[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("{} (validation)".format(name))
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "{}_{}.png".format(name, mark)), bbox_inches = "tight", dpi = 300)

    def test(self, mark = "test"):
        if mark == "test":
            dataloader = self.test_loader
        elif mark == "val":
            dataloader = self.val_loader
        val_data, val_preds, val_loss, val_auc = {}, {}, 0, []
        self.model.eval()
        for data in dataloader:
            loss, val_pred = self.val_on_step(data)
            val_loss += loss
            val_preds = self._update_total_dict(val_preds, val_pred)
            val_data = self._update_total_dict(val_data, data)
        val_loss /= len(dataloader)
        val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, pred_neg_nums, thresholds = self._cal_auc_auprc_metrics(val_preds, val_data)

        # self.draw_preds(val_preds, val_data, mark, thresholds)

        self._draw_performances(val_nums, val_aucs, "auc", mark)
        self._draw_performances(val_nums, val_auprcs, "auprc", mark)
        self._draw_performances(val_nums, val_precisions, "precision", mark)
        self._draw_performances(val_nums, val_recalls, "recall", mark)
        self._draw_performances(val_nums, val_specificities, "specificity", mark)
        self._draw_performances(val_nums, val_f1s, "f1", mark)
        self._draw_performances(val_nums, val_accs, "acc", mark)

        avg_auc = np.mean(list(val_aucs.values()))
        avg_auprc = np.mean(list(val_auprcs.values()))
        avg_precision = np.mean(list(val_precisions.values()))
        avg_recall = np.mean(list(val_recalls.values()))
        avg_specificity = np.mean(list(val_specificities.values()))
        avg_f1 = np.mean(list(val_f1s.values()))
        avg_acc = np.mean(list(val_accs.values()))
        pos_num_df = pd.DataFrame.from_dict(val_nums, orient = "index", columns = ["pos_ratio"]).reset_index().rename(columns = {"index": "odor"})
        pred_neg_df = pd.DataFrame.from_dict(pred_neg_nums, orient = "index", columns = ["pred_neg_ratio"]).reset_index().rename(columns = {"index": "odor"})
        merged = pos_num_df.merge(pred_neg_df, how = "left", on = "odor")
        merged.to_excel(os.path.join(self.out_dir, "pred_summary.xlsx"), index = False)
        return val_loss, avg_auc, avg_auprc, avg_precision, avg_recall, avg_specificity, avg_f1, avg_acc  

    def _sigmoid_np(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))

    def _minmax_np(self, x):
        min_ = np.min(x)
        max_ = np.max(x)
        return (x-min_) / (max_ - min_)

    def save_pred(self, mark = "train", which_emb = "beforeNN"):
        os.makedirs(os.path.join(self.out_dir, "pred_emb"), exist_ok = True)
        if mark == "test":
            dataloader = self.test_loader
        elif mark == "val":
            dataloader = self.val_loader
        elif mark == "train":
            dataloader = self.train_loader
        val_data, val_preds, val_embs, val_smiles = {}, {}, [], []
        self.model.eval()
        for data in tqdm(dataloader, ncols = 80):
            atom_feats, cou_feats, mol_masks, eigval_coulomb, eigvec_coulomb = self._data_to_var(data)
            if which_emb == "afterNN":
                mol_embs = self.model.forward_emb(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)
            elif which_emb == "beforeNN":
                mol_embs, _ = self.model.encoder(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)

            preds = self.model(atom_feats, cou_feats, eigvec_coulomb, eigval_coulomb, mol_masks)

            if len(atom_feats) == 1:
                val_embs.append(self._to_np(mol_embs))
            else:
                val_embs += self._to_np(mol_embs).tolist()

            val_data = self._update_total_dict(val_data, data)
            val_preds = self._update_total_dict(val_preds, preds)
            val_smiles.extend(list(data["smiles"]))
        val_embs = np.array(val_embs)
        df = pd.DataFrame(val_embs, columns = ["emb{}".format(i) for i in range(val_embs.shape[1])])
        for l in self.labels_list:
            if l in val_data:
                true = val_data[l]
                df[l] = true
                # df["{}_pred".format(l)] = self._minmax_np(val_preds[l])
                df["{}_pred".format(l)] = self._sigmoid_np(val_preds[l])
        df["smiles"] = val_smiles
        os.makedirs(os.path.join(self.out_dir, "pred_emb_score_sigmoid"), exist_ok = True)
        df.to_csv(os.path.join(self.out_dir, "pred_emb_score_sigmoid", "{}_{}.csv".format(mark, which_emb)), index = False)


