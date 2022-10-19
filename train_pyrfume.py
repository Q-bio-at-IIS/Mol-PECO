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
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from train_loader import PyrfumeData, DreamTrainData
from test_loader import DreamTestData
from trainer import Trainer, TrainerCoulomb, TrainerAdjCoulomb
from models import GNN
from models_adj_cou import GNNAdjCou
from tester import Tester, TesterCoulomb, TesterAdjCoulomb
from ml_utils import *
from dl_utils import load_pretrained_infograph

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "../dream_pyrfume_models2/dumped30_coulomb_zscore/", type = str)
parser.add_argument("--out_dir", default = "../dream_pyrfume_models2/GNN_addHs5_dumped30_couzscore/", type = str)
parser.add_argument("--gnn_matrix", default = "coulomb") ##adjacent, coulomb, both
parser.add_argument("--coulomb_bin_step", default = -1, type = int)

parser.add_argument("--batch_size", default = 1024, type = int)
parser.add_argument("--epoch", default = 600, type = int)
parser.add_argument("--emb_dim", default = 32, type = int)
parser.add_argument("--hid_layer", default = 4, type = int)
parser.add_argument("--hid_dim", default = 16, type = int)
parser.add_argument("--dropout", default = 0.1, type = float)
parser.add_argument("--lap_dim", default = 16, type = int)
parser.add_argument("--max_freq", default = 5, type = int)
parser.add_argument("--atom_num", default = 17, type = int)
parser.add_argument("--hid_dims", default = []) #[32]
parser.add_argument("--fc_dims", default = [])#[32, 16]
parser.add_argument("--lr", default = 0.001, type = float)

parser.add_argument("--device", default = 1, type = int)
parser.add_argument("--model", default = "GNN", type = str)
parser.add_argument("--out_name", default = "DREAM_EMB", type = str)
parser.add_argument("--method", default = "mean", type = str)
parser.add_argument("--wanted_label", default = "all")

# parser.add_argument("--pretrained_path", default = "../model_pretrain_by_infograph/zinc2m_aicrowd_dream_new2/lr0.001_regu1.0/drop0.1_emb32_hid16_layer8/min.ckpt")
parser.add_argument("--pretrained_path", default = "")
args = parser.parse_args()
torch.cuda.set_device(args.device)
result_path = os.path.join(args.out_dir, "results_zscore.csv")
args.hid_dims = [args.hid_dim] * args.hid_layer
args.fc_dims = [sum(args.hid_dims), 32]
args.out_dir = os.path.join(
    args.out_dir, 
    "lr{}_layer{}".format(
        args.lr, args.hid_layer))

train_data = PyrfumeData(args.data_dir, "pyrfume_train.xlsx")
train_loader = DataLoader(train_data, batch_size = args.batch_size, collate_fn = train_data.collate_fn, shuffle = True)
val_data = PyrfumeData(args.data_dir, "pyrfume_test.xlsx")
val_loader = DataLoader(val_data, batch_size = args.batch_size, collate_fn = val_data.collate_fn, shuffle = False)
if args.model == "GNN":
    if args.gnn_matrix == "adjacent":
        model = GNN(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels, bonds = [0, 1, 2, 3])
    elif args.gnn_matrix == "coulomb":
        if args.coulomb_bin_step == -1:
            bonds = ["coulomb"]
        else:
            bonds = []
            for i in range(1, args.coulomb_bin_step):
                # bonds.append("coulomb{}".format(i))
                bonds.append("coulomb{}".format(-i))
        model = GNN(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels, bonds = bonds)
    elif args.gnn_matrix == "both":
        bonds_adj = [0, 1, 2, 3]
        if args.coulomb_bin_step == -1:
            bonds_cou = ["coulomb"]
        else:
            bonds_cou = []
            for i in range(1, args.coulomb_bin_step):
                # bonds.append("coulomb{}".format(i))
                bonds_cou.append("coulomb{}".format(-i))
        model = GNNAdjCou(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, train_data.labels, 
            bonds_adj = bonds_adj, bonds_cou = bonds_cou)

if args.pretrained_path != "":
    model = load_pretrained_infograph(model, args.pretrained_path)

if args.gnn_matrix == "adjacent":
    trainer = Trainer(model, train_loader, val_loader, args.epoch, train_data.labels, args.out_dir, args.lr)
elif args.gnn_matrix == "coulomb":
    trainer = TrainerCoulomb(model, train_loader, val_loader, args.epoch, train_data.labels, args.out_dir, args.lr)
elif args.gnn_matrix == "both":
    trainer = TrainerAdjCoulomb(model, train_loader, val_loader, args.epoch, train_data.labels, args.out_dir, args.lr)

trainer.fit_classification()

## test on the validation of AIcrowd
model.load_checkpoint(os.path.join(args.out_dir, "max_auc.ckpt"))
val_loss, val_auc = trainer.test()

## test on the DREAM data
def gen_mol_emb(model, args, dataset, marker):
    data_loader = DataLoader(dataset, batch_size = args.batch_size, collate_fn = dataset.collate_fn, shuffle = True)
    if args.gnn_matrix == "adjacent":
        tester = Tester(model, data_loader, dataset.labels, args.out_dir)
    elif args.gnn_matrix == "coulomb":
        tester = TesterCoulomb(model, data_loader, dataset.labels, args.out_dir)
    elif args.gnn_matrix == "both":
        tester = TesterAdjCoulomb(model, data_loader, dataset.labels, args.out_dir)

    mol_emb = tester.generate_emb(marker)
    return mol_emb

# generate the molecular embedding
for dilution in ["all", "10", "100000"]:
    for dream_norm in ["none", "zscore"]:
        train_data = DreamTrainData(args.data_dir, dilution, args.method, args.wanted_label, dream_norm)
        train_emb = gen_mol_emb(model, args, train_data, "train")
        val_data = DreamTestData(args.data_dir, "val", dilution, args.wanted_label, dream_norm)
        val_data.set_mus_sigmas(train_data.mus, train_data.sigmas)
        val_emb = gen_mol_emb(model, args, val_data, "val")
        test_data = DreamTestData(args.data_dir, "test", dilution, args.wanted_label, dream_norm)
        test_data.set_mus_sigmas(train_data.mus, train_data.sigmas)
        test_emb = gen_mol_emb(model, args, test_data, "test")
        tot_emb = pd.concat([train_emb, val_emb, test_emb])

        out_dir = os.path.join(args.out_dir, "{}_dilu{}_nonzero_{}".format(args.out_name, dilution, dream_norm))
        os.makedirs(out_dir, exist_ok = True)
        tot_emb.to_excel(os.path.join(out_dir, "pred_emb.xlsx"), index = False)

        test_rs, test_ps = [], []
        for l in test_data.labels:
            test_r, test_p = finetune_rf(tot_emb, l, out_dir)
            test_rs.append(test_r)
            test_ps.append(test_p)
        dream_df = pd.DataFrame()
        dream_df["label"] = test_data.labels
        dream_df["r"] = test_rs
        dream_df["p"] = test_ps
        dream_df.to_excel(os.path.join(out_dir, "pearson_{}.xlsx".format(dilution)))
        print(dilution, np.mean(test_rs))
        with open(result_path, "a+") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                args.model, args.lr, args.dropout, args.emb_dim, args.hid_dim, args.lap_dim, args.hid_layer, args.max_freq,
                val_loss, val_auc, dilution, dream_norm, np.mean(test_rs)))


