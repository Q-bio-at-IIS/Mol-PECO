#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-16 15:53:25
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$
import os, torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

class MLP(nn.Module):
    """docstring for ClassName"""
    def __init__(
        self, in_dim, fc_dims, drop_rate, label_names):
        super(MLP, self).__init__()
        self.label_names = label_names

        self.fc_layers = nn.ModuleList()
        in_dims = [in_dim] + fc_dims
        out_dims = fc_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            fc_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(drop_rate)
                )
            self.fc_layers.append(fc_layer)

        self.out_layers = {}
        for l in label_names:
            out_layer = nn.Linear(fc_dims[-1], 1)
            self.out_layers[l] = out_layer
        self._add_modules(self.out_layers, "fc")

    def _add_modules(self, layers, n):
        ##add modules
        modules = {}
        modules.update(layers)
        for k, v in modules.items():
            name = "{}_{}".format(n, k)
            self.add_module(name, v)      

    def forward(self, mol_embs):
        for fc in self.fc_layers:
            mol_embs = fc(mol_embs)

        preds = {}
        for l in self.label_names:
            pred = self.out_layers[l](mol_embs)
            preds[l] = pred
        return preds

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class EmbData(Dataset):
    """docstring for EmbData"""
    def __init__(self, xs, ys, labels):
        super(EmbData, self).__init__()
        self.xs = xs
        self.ys = ys
        self.labels = labels
        self.label_weights = {}
        for i, l in enumerate(self.labels):
            cnt = 0
            this_ys = ys[:, i]
            ratio = sum(this_ys)/len(this_ys)
            weight = 1 - ratio
            self.label_weights[l] = weight

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        data = []
        data.extend(list(self.xs[idx]))
        data.extend(list(self.ys[idx]))
        return data

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
        xs, ys = data[:, :-len(self.labels)], data[:, -len(self.labels):]
        data_dict = {}
        for i, l in enumerate(self.labels):
            data_dict[l], data_dict["{}_weight".format(l)] = self._pro_label(ys, i)

        data_dict["xs"] = xs
        data_dict["label_weights"] = self.label_weights
        return data_dict

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, model, train_loader, val_loader, test_loader, epoch, labels, out_dir, lr):
        super(Trainer, self).__init__()
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epoch = epoch
        self.labels_list = labels
        self.out_dir = out_dir
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr = lr/10., steps_per_epoch = len(self.train_loader), epochs = self.epoch)
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

    def _cal_auc_auprc_metrics(self, preds, labels):
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for l in tqdm(self.labels_list, ncols = 80):
            if sum(labels[l]) == 0:
                continue
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

    def train_on_step(self, data):
        xs = data["xs"]
        xs = self._to_var(xs, t = "float")
        self.optimizer.zero_grad()
        preds = self.model(xs)
        loss = self._call_loss_log(preds, data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def val_on_step(self, data):
        xs = data["xs"]
        xs = self._to_var(xs, t = "float")
        preds = self.model(xs)
        loss = self._call_loss_log(preds, data)
        return loss.item(), preds

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
                if len(data["xs"]) == 1:
                    continue
                total_loss = self.train_on_step(data)
                train_loss_tot += total_loss
            train_loss_tot /= len(self.train_loader)

            self.model.eval()
            for data in self.val_loader:
                total_loss, val_pred = self.val_on_step(data)
                val_loss_tot += total_loss
                val_preds = self._update_total_dict(val_preds, val_pred)
                val_data = self._update_total_dict(val_data, data)
            val_loss_tot /= len(self.val_loader)

            val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, _ = self._cal_auc_auprc_metrics(val_preds, val_data)

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
        val_aucs, val_auprcs, val_precisions, val_recalls, val_specificities, val_f1s, val_accs, val_nums, thresholds = self._cal_auc_auprc_metrics(val_preds, val_data)

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

