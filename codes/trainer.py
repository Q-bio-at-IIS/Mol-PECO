#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-17 12:53:17
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import numpy as np
import pandas as pd
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

class TrainerBase(object):
    """docstring for TrainerBase"""
    def __init__(self, model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr):
        super(TrainerBase, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epoch = epoch
        self.labels_list = labels
        self.out_dir = out_dir
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = lr)
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
            # print(np.min(labels["{}_weight".format(l)].squeeze()), np.max(labels["{}_weight".format(l)].squeeze()))
            weight = self._to_var(labels["{}_weight".format(l)].squeeze())
            pred, label = preds[l], self._to_var(labels[l])
            # print(l, torch.min(pred), torch.max(pred), torch.min(label), torch.max(label))
            pred = torch.sigmoid(pred)
            # print(l, torch.min(pred), torch.max(pred), torch.min(label), torch.max(label))
            mae = torch.mean(self.criterion(pred, label)*weight)
            log = torch.abs(torch.log(pred + epsilon) - torch.log(label + epsilon))
            mean_log = torch.mean(log*weight)
            this_loss = mae + mean_log
            this_loss *= labels["label_weights"][l]

            losses += this_loss
        # print("=="*20)
        return losses/len(self.labels_list)

    def _cal_auc_auprc_metrics(self, preds, labels):
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for l in self.labels_list:
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
            auprcs[l] = auprc
            precisions[l] = precision
            recalls[l] = recall
            specificities[l] = specificity
            f1s[l] = f1
            accs[l] = acc
            thresholds[l] = threshold

        return aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds

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



    def save_model(self, epoch, name):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "{}.ckpt".format(name)))

    def _update_total_dict(self, total_dict, dicts):
        for l in self.labels_list:
            if l in total_dict:
                total_dict[l] = total_dict[l] + self._to_np(dicts[l]).tolist()
            else:
                total_dict[l] = self._to_np(dicts[l]).tolist()
        return total_dict

    def fit_classification(self):
        best_loss, best_auc = 9999, -1
        e_auc, e_loss = 0, 0
        pbar = tqdm(range(self.epoch), ncols = 120)
        for e in pbar:
            train_loss, val_data, val_preds, val_loss, val_auc = 0, {}, {}, 0, []

            self.model.train()
            for data in self.train_loader:
                loss, preds = self.train_on_step(data)
                train_loss += loss
            train_loss /= len(self.train_loader)

            self.model.eval()
            for data in self.val_loader:
                loss, val_pred = self.val_on_step(data)
                val_loss += loss
                val_preds = self._update_total_dict(val_preds, val_pred)
                val_data = self._update_total_dict(val_data, data)
            val_loss /= len(self.val_loader)
            val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, _ = self._cal_auc_auprc_metrics(val_preds, val_data)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": val_loss}, e)
            self.writer.add_scalars("AUC (validation)", val_aucs, e)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], e)
            if val_loss < best_loss:
                self.save_model(e, "min_loss")
                best_loss = val_loss
                e_loss = e
                pbar.set_description("min loss at {}: {}\tmax auc at {}: {}".format(e_loss, round(best_loss, 2), e_auc, round(best_auc, 2)))

            if np.mean(list(val_aucs.values())) > best_auc:
                best_auc = np.mean(list(val_aucs.values()))
                e_auc = e
                self.save_model(e, "max_auc")
                pbar.set_description("min loss at {}: {}\tmax auc at {}: {}".format(e_loss, round(best_loss, 2), e_auc, round(best_auc, 2)))

        return best_loss, best_auc       

    def _draw_performances(self, nums, metrics, name, mark):
        outs = []
        fig, ax = plt.subplots()
        for l in self.labels_list:
            if l in metrics:
                ax.scatter(nums[l], metrics[l], label = l)
                outs.append([l, nums[l], metrics[l]])
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("{} (validation)".format(name))
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "{}_{}.png".format(name, mark)), bbox_inches = "tight", dpi = 300)

        df = pd.DataFrame(outs, columns = ["label", "positive num", "metrics"])
        df.to_excel(os.path.join(self.out_dir, "{}_{}.xlsx".format(name, mark)))

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
        val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, thresholds = self._cal_auc_auprc_metrics(val_preds, val_data)

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
        return val_loss, avg_auc, avg_auprc, avg_precision, avg_recall, avg_specificity, avg_f1, avg_acc

    def _data_to_var(self, data):
        pass

    def train_on_step(self, data):
        pass

    def val_on_step(self, data):
        pass

class Trainer(TrainerBase):
    """docstring for Trainer"""
    def __init__(self, model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr):
        super(Trainer, self).__init__(model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr)

    def _data_to_var(self, data):
        atom_feats, adj_feats, mol_masks = data["atom_feat"], data["adj_feat"], data["mol_mask"]
        atom_feats = self._to_var(atom_feats, t = "int")
        adj_feats = self._to_var_dict(adj_feats)
        mol_masks = self._to_var(mol_masks, t = "bool")
        return atom_feats, adj_feats, mol_masks

    def train_on_step(self, data):
        atom_feats, adj_feats, mol_masks = self._data_to_var(data)
        self.optimizer.zero_grad()
        preds = self.model(atom_feats, adj_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), preds

    def val_on_step(self, data):
        atom_feats, adj_feats, mol_masks = self._data_to_var(data)
        preds = self.model(atom_feats, adj_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        return loss.item(), preds                        
        
class TrainerCoulomb(TrainerBase):
    """docstring for TrainerCoulomb"""
    def __init__(self, model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr):
        super(TrainerCoulomb, self).__init__(model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr)

    def _data_to_var(self, data):
        atom_feats, cou_feats, mol_masks = data["atom_feat"], data["coulomb_feat"], data["mol_mask"]
        atom_feats = self._to_var(atom_feats, t = "int")
        cou_feats = self._to_var_dict(cou_feats)
        mol_masks = self._to_var(mol_masks, t = "bool")
        return atom_feats, cou_feats, mol_masks

    def train_on_step(self, data):
        atom_feats, cou_feats, mol_masks = self._data_to_var(data)
        self.optimizer.zero_grad()
        preds = self.model(atom_feats, cou_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), preds

    def val_on_step(self, data):
        atom_feats, cou_feats, mol_masks = self._data_to_var(data)
        preds = self.model(atom_feats, cou_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        return loss.item(), preds
