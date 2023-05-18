#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-09-26 16:22:17
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, argparse, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from train_loader import PyrfumeData
from trainer import Trainer, TrainerCoulomb
from models import GNN
from tester import Tester, TesterCoulomb
from ml_utils import *
from dl_utils import load_pretrained_infograph

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "../data/", type = str)
parser.add_argument("--out_dir", default = "../GNN-Coulomb/", type = str)
parser.add_argument("--gnn_matrix", default = "coulomb") ##adjacent, coulomb, both
parser.add_argument("--coulomb_bin_step", default = -1, type = int)

parser.add_argument("--batch_size", default = 1024, type = int)
parser.add_argument("--epoch", default = 600, type = int)
parser.add_argument("--emb_dim", default = 32, type = int)
parser.add_argument("--hid_layer", default = 8, type = int)
parser.add_argument("--hid_dim", default = 16, type = int)
parser.add_argument("--dropout", default = 0.1, type = float)
parser.add_argument("--lap_dim", default = 16, type = int)
parser.add_argument("--max_freq", default = 20, type = int)
parser.add_argument("--atom_num", default = 17, type = int)
parser.add_argument("--hid_dims", default = []) #[32]
parser.add_argument("--fc_dims", default = []) #[32, 16]
parser.add_argument("--lr", default = 1, type = float)

parser.add_argument("--device", default = 1, type = int)
parser.add_argument("--model", default = "gcn", type = str)

parser.add_argument("--pretrained_path", default = "")
args = parser.parse_args()
torch.cuda.set_device(args.device)
result_path = os.path.join(args.out_dir, "results_min_loss.csv")
args.hid_dims = [args.hid_dim] * args.hid_layer
args.fc_dims = [sum(args.hid_dims), 32]
args.out_dir = os.path.join(
    args.out_dir, 
    "lr{}_layer{}".format(
        args.lr, args.hid_layer))

train_data = PyrfumeData(args.data_dir, "pyrfume_train.xlsx")
train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, shuffle = True)
val_data = PyrfumeData(args.data_dir, "pyrfume_val.xlsx")
val_loader = DataLoader(val_data, batch_size = args.batch_size, collate_fn = val_data.collate_fn, shuffle = False)
test_data = PyrfumeData(args.data_dir, "pyrfume_test.xlsx")
test_loader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = test_data.collate_fn, shuffle = False)

if args.gnn_matrix == "adjacent":
    model = GNN(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels, bonds = [0, 1, 2, 3], gnn_name = args.model)
elif args.gnn_matrix == "coulomb":
    if args.coulomb_bin_step == -1:
        bonds = ["coulomb"]
    else:
        bonds = []
        for i in range(1, args.coulomb_bin_step):
            # bonds.append("coulomb{}".format(i))
            bonds.append("coulomb{}".format(-i))
    model = GNN(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels, bonds = bonds, gnn_name = args.model)

if args.pretrained_path != "":
    model = load_pretrained_infograph(model, args.pretrained_path)

if args.gnn_matrix == "adjacent":
    trainer = Trainer(model, train_loader, val_loader, test_loader, args.epoch, train_data.labels, args.out_dir, args.lr)
elif args.gnn_matrix == "coulomb":
    trainer = TrainerCoulomb(model, train_loader, val_loader, test_loader, args.epoch, train_data.labels, args.out_dir, args.lr)

trainer.fit_classification()

## test on the validation  
model.load_checkpoint(os.path.join(args.out_dir, "min_loss.ckpt"))

val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc = trainer.test("val")
test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc = trainer.test("test")

with open(result_path, "a+") as f:
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                args.model, args.lr, args.dropout, args.emb_dim, args.hid_dim, args.hid_layer, 
                val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc,
                test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc))

