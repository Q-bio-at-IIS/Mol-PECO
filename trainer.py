#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-17 12:53:17
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import numpy as np
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

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model, train_loader, val_loader, epoch, labels, out_dir, lr):
        super(Trainer, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
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

    # def _call_loss(self, preds, labels):
    #     losses = 0
    #     for l in self.labels_list:
    #         # loss = self.criterion(F.sigmoid(preds[l]), self._to_var(labels[l]))
    #         loss = self.criterion(preds[l], self._to_var(labels[l]))
    #         weight = self._to_var(labels["{}_weight".format(l)].squeeze(), "bool")
    #         loss = torch.mean(loss * weight)
    #         losses += loss
    #     return losses/len(self.labels_list)

    def _call_loss_log(self, preds, labels, epsilon = 1e-9):
        losses = 0 
        for l in self.labels_list:
            weight = self._to_var(labels["{}_weight".format(l)].squeeze())
            pred, label = preds[l], self._to_var(labels[l])
            pred = torch.sigmoid(pred)
            mae = torch.mean(self.criterion(pred, label)*weight)
            log = torch.abs(torch.log(pred + epsilon) - torch.log(label + epsilon))
            mean_log = torch.mean(log*weight)
            this_loss = mae + mean_log
            this_loss *= labels["label_weights"][l]

            losses += this_loss

        return losses/len(self.labels_list)

    def _cal_pearson(self, preds, labels):
        rs = []
        for l in self.labels_list:
            r, p = pearsonr(self._to_np(preds[l]), labels[l].squeeze())
            rs.append(r)
        return rs

    def _cal_auc(self, preds, labels):
        aucs, nums = {}, {}
        for l in self.labels_list:
            auc = metrics.roc_auc_score(labels[l], preds[l])
            aucs[l] = auc
            nums[l] = sum(labels[l])/len(labels[l])

        return aucs, nums

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
            val_aucs, val_nums = self._cal_auc(val_preds, val_data)

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

        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)
        return best_loss, best_auc       

    def test(self):
        val_data, val_preds, val_loss, val_auc = {}, {}, 0, []
        self.model.eval()
        for data in self.val_loader:
            loss, val_pred = self.val_on_step(data)
            val_loss += loss
            val_preds = self._update_total_dict(val_preds, val_pred)
            val_data = self._update_total_dict(val_data, data)
        val_loss /= len(self.val_loader)
        val_aucs, val_nums = self._cal_auc(val_preds, val_data)
        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)

        return val_loss, np.mean(list(val_aucs.values())) 
        
class Trainer1Out(object):
    """docstring for Trainer"""
    def __init__(self, model, train_loader, val_loader, epoch, labels, out_dir, lr):
        super(Trainer1Out, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = epoch
        self.label_names = labels
        self.out_dir = out_dir
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr = lr/10., steps_per_epoch = len(self.train_loader), epochs = self.epoch)
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(os.path.join(out_dir, "logging"))
        os.makedirs(self.out_dir, exist_ok = True)

    def _to_var_dict(self, x):
        new_dict = {}
        for k, v in x.items():
            new_dict[k] = self._to_var(v, t = "float")
        return new_dict

    def _to_var(self, x, t = "float"):
        x = np.array(x)
        if t == "int":
            return torch.LongTensor(x).cuda()
        elif t == "bool":
            return torch.BoolTensor(x).cuda()
        else:
            return torch.FloatTensor(x).cuda()

    def _to_np(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().squeeze()
        else:
            return x.squeeze()

    def _call_loss(self, preds, labels):
        loss = self.criterion(preds, self._to_var(labels, t = "int"))
        return loss

    def _cal_auc(self, preds, labels):
        labels_set = sorted(set(labels))

        aucs = {}
        labels = np.array(labels)
        for l in labels_set:
            this_preds = preds[:, l]
            this_labels = labels == l
            auc = metrics.roc_auc_score(this_labels, this_preds)
            aucs[self.label_names[l]] = auc
        return aucs

    def _data_to_var(self, data):
        atom_feats, adj_feats = data["atom_feat"], data["adj_feat"]
        eigval_adj, eigvec_adj = data["eigval_adj"], data["eigvec_adj"]
        atom_feats = self._to_var(atom_feats, t = "int")
        adj_feats = self._to_var_dict(adj_feats)
        eigval_adj = self._to_var(eigval_adj, t = "float")
        eigvec_adj = self._to_var(eigvec_adj, t = "float")
        return atom_feats, adj_feats, eigval_adj, eigvec_adj

    def train_on_step(self, data):
        atom_feats, adj_feats, eigval_adj, eigvec_adj = self._data_to_var(data)
        self.optimizer.zero_grad()
        preds = self.model(atom_feats, adj_feats, eigval_adj, eigvec_adj)
        loss = self._call_loss(preds, data["labels"])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), preds

    def val_on_step(self, data):
        atom_feats, adj_feats, eigval_adj, eigvec_adj = self._data_to_var(data)
        preds = self.model(atom_feats, adj_feats, eigval_adj, eigvec_adj)
        loss = self._call_loss(preds, data["labels"])
        return loss.item(), preds

    def save_model(self, epoch, name):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "{}.ckpt".format(name)))

    def fit_classification(self):
        best_loss, best_auc = 9999, -1
        e_auc, e_loss = 0, 0
        pbar = tqdm(range(self.epoch), ncols = 120)
        for e in pbar:
            train_loss, val_labels, val_preds, val_loss, val_auc = 0, [], [], 0, []

            self.model.train()
            for data in self.train_loader:
                loss, preds = self.train_on_step(data)
                train_loss += loss
            train_loss /= len(self.train_loader)

            self.model.eval()

            for data in self.val_loader:
                loss, val_pred = self.val_on_step(data)
                val_loss += loss
                val_preds.append(val_pred.detach().cpu().numpy())
                val_labels.extend(list(data["labels"]))
            val_preds = np.vstack(val_preds)

            val_loss /= len(self.val_loader)
            val_aucs = self._cal_auc(val_preds, val_labels)
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

class TrainerCoulomb1Out(object):
    """docstring for Trainer"""
    def __init__(self, model, train_loader, val_loader, epoch, labels, out_dir, lr, method = "zscore"):
        super(TrainerCoulomb1Out, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = epoch
        self.label_names = labels
        self.out_dir = out_dir
        self.method = method
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr = lr/10., steps_per_epoch = len(self.train_loader), epochs = self.epoch)
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(os.path.join(out_dir, "logging"))
        os.makedirs(self.out_dir, exist_ok = True)

    def _to_var_dict(self, x):
        new_dict = {}
        for k, v in x.items():
            new_dict[k] = self._to_var(v, t = "float")
        return new_dict

    def _to_var(self, x, t = "float"):
        x = np.array(x)
        if t == "int":
            return torch.LongTensor(x).cuda()
        elif t == "bool":
            return torch.BoolTensor(x).cuda()
        else:
            return torch.FloatTensor(x).cuda()

    def _to_np(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().squeeze()
        else:
            return x.squeeze()

    def _call_loss(self, preds, labels):
        loss = self.criterion(preds, self._to_var(labels, t = "int"))
        return loss

    def _cal_auc(self, preds, labels):
        labels_set = sorted(set(labels))

        aucs = {}
        labels = np.array(labels)
        for l in labels_set:
            this_preds = preds[:, l]
            this_labels = labels == l
            auc = metrics.roc_auc_score(this_labels, this_preds)
            aucs[self.label_names[l]] = auc
        return aucs

    def _data_to_var(self, data):
        atom_feats, coulomb_feats = data["atom_feat"], data["coulomb_feat"]
        eigval_coulomb, eigvec_coulomb = data["eigval_coulomb"], data["eigvec_coulomb"]
        mol_masks = data["mol_mask"]
        atom_feats = self._to_var(atom_feats, t = "int")
        coulomb_feats = {"coulomb": self._to_var(coulomb_feats, t = "float")}
        eigval_coulomb = self._to_var(eigval_coulomb, t = "float")
        eigvec_coulomb = self._to_var(eigvec_coulomb, t = "float")
        mol_masks = self._to_var(mol_masks, t = "bool")
        return atom_feats, coulomb_feats, eigval_coulomb, eigvec_coulomb, mol_masks

    def train_on_step(self, data):
        atom_feats, coulomb_feats, eigval_coulomb, eigvec_coulomb, mol_masks = self._data_to_var(data)
        self.optimizer.zero_grad()
        preds = self.model(atom_feats, coulomb_feats, mol_masks)
        loss = self._call_loss(preds, data["labels"])
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), preds

    def val_on_step(self, data):
        atom_feats, coulomb_feats, eigval_coulomb, eigvec_coulomb, mol_masks = self._data_to_var(data)
        preds = self.model(atom_feats, coulomb_feats, mol_masks)
        loss = self._call_loss(preds, data["labels"])
        return loss.item(), preds

    def save_model(self, epoch, name):
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, "{}.ckpt".format(name)))

    def fit_classification(self):
        best_loss, best_auc = 9999, -1
        e_auc, e_loss = 0, 0
        pbar = tqdm(range(self.epoch), ncols = 120)
        for e in pbar:
            train_loss, val_labels, val_preds, val_loss, val_auc = 0, [], [], 0, []

            self.model.train()
            for data in self.train_loader:
                loss, preds = self.train_on_step(data)
                train_loss += loss
            train_loss /= len(self.train_loader)

            self.model.eval()

            for data in self.val_loader:
                loss, val_pred = self.val_on_step(data)
                val_loss += loss
                val_preds.append(val_pred.detach().cpu().numpy())
                val_labels.extend(list(data["labels"]))
            val_preds = np.vstack(val_preds)

            val_loss /= len(self.val_loader)
            val_aucs = self._cal_auc(val_preds, val_labels)
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

class TrainerCoulomb(object):
    """docstring for TrainerCoulomb"""
    def __init__(self, model, train_loader, val_loader, epoch, labels, out_dir, lr):
        super(TrainerCoulomb, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
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
            weight = self._to_var(labels["{}_weight".format(l)].squeeze())
            pred, label = preds[l], self._to_var(labels[l])
            pred = torch.sigmoid(pred)
            mae = torch.mean(self.criterion(pred, label)*weight)
            log = torch.abs(torch.log(pred + epsilon) - torch.log(label + epsilon))
            mean_log = torch.mean(log*weight)
            this_loss = mae + mean_log
            this_loss *= labels["label_weights"][l]

            losses += this_loss

        return losses/len(self.labels_list)

    def _cal_auc(self, preds, labels):
        aucs, nums = {}, {}
        for l in self.labels_list:
            auc = metrics.roc_auc_score(labels[l], preds[l])
            aucs[l] = auc
            nums[l] = sum(labels[l])/len(labels[l])

        return aucs, nums

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
            val_aucs, val_nums = self._cal_auc(val_preds, val_data)

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

        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)
        return best_loss, best_auc

    def test(self):
        val_data, val_preds, val_loss, val_auc = {}, {}, 0, []
        self.model.eval()
        for data in self.val_loader:
            loss, val_pred = self.val_on_step(data)
            val_loss += loss
            val_preds = self._update_total_dict(val_preds, val_pred)
            val_data = self._update_total_dict(val_data, data)
        val_loss /= len(self.val_loader)
        val_aucs, val_nums = self._cal_auc(val_preds, val_data)
        
        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)

        return val_loss, np.mean(list(val_aucs.values()))   
        
class TrainerAdjCoulomb(object):
    """docstring for TrainerCoulomb"""
    def __init__(self, model, train_loader, val_loader, epoch, labels, out_dir, lr):
        super(TrainerAdjCoulomb, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
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
            weight = self._to_var(labels["{}_weight".format(l)].squeeze())
            pred, label = preds[l], self._to_var(labels[l])
            pred = torch.sigmoid(pred)
            mae = torch.mean(self.criterion(pred, label)*weight)
            log = torch.abs(torch.log(pred + epsilon) - torch.log(label + epsilon))
            mean_log = torch.mean(log*weight)
            this_loss = mae + mean_log
            this_loss *= labels["label_weights"][l]

            losses += this_loss

        return losses/len(self.labels_list)

    def _cal_auc(self, preds, labels):
        aucs, nums = {}, {}
        for l in self.labels_list:
            auc = metrics.roc_auc_score(labels[l], preds[l])
            aucs[l] = auc
            nums[l] = sum(labels[l])/len(labels[l])

        return aucs, nums

    def _data_to_var(self, data):
        atom_feats, adj_feats, cou_feats, mol_masks = data["atom_feat"], data["adj_feat"], data["coulomb_feat"], data["mol_mask"]
        atom_feats = self._to_var(atom_feats, t = "int")
        adj_feats = self._to_var_dict(adj_feats)
        cou_feats = self._to_var_dict(cou_feats)
        mol_masks = self._to_var(mol_masks, t = "bool")
        return atom_feats, adj_feats, cou_feats, mol_masks

    def train_on_step(self, data):
        atom_feats, adj_feats, cou_feats, mol_masks = self._data_to_var(data)
        self.optimizer.zero_grad()
        preds = self.model(atom_feats, adj_feats, cou_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), preds

    def val_on_step(self, data):
        atom_feats, adj_feats, cou_feats, mol_masks = self._data_to_var(data)
        preds = self.model(atom_feats, adj_feats, cou_feats, mol_masks)
        loss = self._call_loss_log(preds, data)
        return loss.item(), preds

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
            val_aucs, val_nums = self._cal_auc(val_preds, val_data)

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

        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)
        return best_loss, best_auc

    def test(self):
        val_data, val_preds, val_loss, val_auc = {}, {}, 0, []
        self.model.eval()
        for data in self.val_loader:
            loss, val_pred = self.val_on_step(data)
            val_loss += loss
            val_preds = self._update_total_dict(val_preds, val_pred)
            val_data = self._update_total_dict(val_data, data)
        val_loss /= len(self.val_loader)
        val_aucs, val_nums = self._cal_auc(val_preds, val_data)

        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.labels_list:
            ax.scatter(val_nums[l], val_aucs[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("AUC (validation)")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "clf.png"), bbox_inches = "tight", dpi = 300)

        return val_loss, np.mean(list(val_aucs.values()))        
      