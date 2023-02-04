#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-03 13:20:07
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, argparse, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from train_loader import PyrfumeData, DreamTrainData
from test_loader import DreamTestData
from trainer_lpe import TrainerCoulomb
from models_lpe import GNNLPE
from tester_lpe import TesterCoulomb
from ml_utils import *
from dl_utils import load_pretrained_infograph

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "../pyrfume_sois_canon/dumped30_coulomb_frobenius/", type = str)
parser.add_argument("--out_dir", default = "../pyrfume_sois_canon/GNNLPE_addHs7_dumped30_coufrobenius/", type = str)
parser.add_argument("--gnn_matrix", default = "coulomb") ##adjacent, coulomb, both
parser.add_argument("--coulomb_bin_step", default = -1)

parser.add_argument("--batch_size", default = 64, type = int)
parser.add_argument("--epoch", default = 600, type = int)
parser.add_argument("--emb_dim", default = 32, type = int)
parser.add_argument("--hid_layer", default = 8, type = int)
parser.add_argument("--hid_dim", default = 16, type = int) ## 16 for GNNLPE_addHs7
parser.add_argument("--dropout", default = 0.1, type = float)
parser.add_argument("--atom_num", default = 17, type = int)
parser.add_argument("--hid_dims", default = []) #[32]
parser.add_argument("--fc_dims", default = [])#[32, 16]
parser.add_argument("--lr", default = 0.01, type = float) ## learning rate of > 1 leading to model collapse

parser.add_argument("--max_freq", default = 20, type = int)
parser.add_argument("--LPE_dim", default = 32, type = int)
parser.add_argument("--LPE_n_heads", default = 4, type = int)
parser.add_argument("--LPE_layer", default = 5, type = int)

parser.add_argument("--device", default = 0, type = int)

parser.add_argument("--pretrained_path", default = "")
args = parser.parse_args()
torch.cuda.set_device(args.device)
result_path = os.path.join(args.out_dir, "results_min_loss.csv")
args.hid_dims = [args.hid_dim] * args.hid_layer
args.fc_dims = [sum(args.hid_dims), 32]


train_data = PyrfumeData(args.data_dir, "pyrfume_train.xlsx")
train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, shuffle = True)
val_data = PyrfumeData(args.data_dir, "pyrfume_val.xlsx")
val_loader = DataLoader(val_data, batch_size = args.batch_size, collate_fn = val_data.collate_fn, shuffle = False)
test_data = PyrfumeData(args.data_dir, "pyrfume_test.xlsx")
test_loader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = test_data.collate_fn, shuffle = False)
if args.gnn_matrix == "coulomb":
    if args.coulomb_bin_step == -1:
        bonds = ["coulomb"]
    else:
        bonds = []
        for i in range(1, args.coulomb_bin_step):
            # bonds.append("coulomb{}".format(i))
            bonds.append("coulomb{}".format(-i))
    model = GNNLPE(
        args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels,
        args.LPE_dim, args.LPE_n_heads, args.LPE_layer, args.max_freq, 
        bonds = bonds)

if args.pretrained_path != "":
    model = load_pretrained_infograph(model, args.pretrained_path)

if args.gnn_matrix == "coulomb":
    trainer = TrainerCoulomb(model, train_loader, val_loader, test_loader, args.epoch, train_data.labels, args.out_dir, args.lr)

trainer.fit_classification()

## evauate on the testing set
model.load_checkpoint(os.path.join(args.out_dir, "min_loss.ckpt"))
val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc = trainer.test("val")
test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc = trainer.test("test")

if not os.path.exists(result_path):
    with open(result_path, "a+") as f:
        f.write("lr,drop,emb,hid_dim,lpe_dim,hid_layer,max_freq,\
            val_loss,val_auc,val_auprc,val_precision,val_recall,val_spe,val_f1,val_acc,\
            test_loss,test_auc,test_auprc,test_precision,test_recall,test_spe,test_f1,test_acc\n")

with open(result_path, "a+") as f:
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                args.lr, args.dropout, 
                args.emb_dim, args.hid_dim, args.LPE_dim, args.hid_layer, args.max_freq,
                val_loss, val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc,
                test_loss, test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc))
