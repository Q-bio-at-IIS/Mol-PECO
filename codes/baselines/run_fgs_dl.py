#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-16 15:50:47
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, argparse, torch
import pandas as pd
from torch.utils.data import DataLoader

from utils import DataPreprocess
from dl_utils import EmbData, MLP, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--fp", default = "deepodor", type = str) ##bfps, cfps, mordreds
parser.add_argument("--data_dir", default = "../../pyrfume_models3_sois/GNNLPE_addHs7_dumped30_coufrobenius/0.0_0.25_0.75/lr0.01_dim16/pred_emb_afterNN/", type = str)
# parser.add_argument("--data_dir", default = "../../pyrfume_models3_sois/dumped30_coulomb_frobenius/", type = str)
parser.add_argument("--label_path", default = "../../pyrfume_models3_sois/dumped30_coulomb_frobenius/labels.xlsx", type = str)
parser.add_argument("--out_dir", default = "../../pyrfume_models3_sois/baselines_dl_multilabel_bs64/", type = str)

parser.add_argument("--batch_size", default = 64, type = int)
parser.add_argument("--epoch", default = 600, type = int)
parser.add_argument("--dropout", default = 0.1, type = float)
parser.add_argument("--fc_dims", default = [32, 32, 1])
parser.add_argument("--lr", default = 0.01, type = float) ## learning rate of > 1 leading to model collapse

parser.add_argument("--device", default = 0, type = int)

args = parser.parse_args()
torch.cuda.set_device(args.device)
result_path = os.path.join(args.out_dir, "results_min_loss.csv")
args.out_dir = os.path.join(args.out_dir, "lr{}_{}".format(args.lr, args.fp))
dp = DataPreprocess(args.data_dir, args.fp)
train_xs, train_ys, val_xs, val_ys, test_xs, test_ys = dp.process()
labels = pd.read_excel(args.label_path)["labels"].values.squeeze().tolist()

train_data = EmbData(train_xs, train_ys, labels)
train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, shuffle = True)
val_data = EmbData(val_xs, val_ys, labels)
val_loader = DataLoader(val_data, batch_size = args.batch_size, collate_fn = val_data.collate_fn, shuffle = False)
test_data = EmbData(test_xs, test_ys, labels)
test_loader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = test_data.collate_fn, shuffle = False)

model = MLP(train_xs.shape[1], args.fc_dims, args.dropout, labels)

trainer = Trainer(model, train_loader, val_loader, test_loader, args.epoch, labels, args.out_dir, args.lr)
trainer.fit_classification()

model.load_checkpoint(os.path.join(args.out_dir, "min_loss.ckpt"))
val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc = trainer.test("val")
test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc = trainer.test("test")

with open(result_path, "a+") as f:
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                args.fp, args.lr, args.dropout, 
                val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc,
                test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc))