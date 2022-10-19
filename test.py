#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-19 13:30:45
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, argparse
from torch.utils.data import DataLoader
from train_loader import DreamTrainData
from test_loader import DreamTestData
from tester import Tester
from models import SimpleNN, GNN

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default = "../data_DREAM", type = str)
parser.add_argument("--model_dir", default = "../models_maskzero_smallerLR_softmax/GNN", type = str)
parser.add_argument("--model_name", default = "min.ckpt", type = str)
parser.add_argument("--batch_size", default = 1024, type = int)
parser.add_argument("--epoch", default = 300, type = int)

parser.add_argument("--atom_num", default = 6, type = int)
parser.add_argument("--emb_dim", default = 32, type = int)
parser.add_argument("--hid_dims", default = [15, 20, 27, 36]) #[32]
parser.add_argument("--fc_dims", default = [98, 64])#[32, 16]
parser.add_argument("--dropout", default = 0.47, type = float)

parser.add_argument("--out_name", default = "test_plots")
parser.add_argument("--dilution", default = "100000", type = str)
parser.add_argument("--wanted_label", default = "all")
parser.add_argument("--method", default = "individual", type = str)

args = parser.parse_args()
args.model_dir = "{}_dilution{}_{}_{}".format(args.model_dir, args.dilution, args.wanted_label, args.method)

test_data = DreamTestData(args.data_dir, "test", args.dilution, args.wanted_label)
test_loader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = test_data.collate_fn, shuffle = False)
# test_data = DreamTrainData(args.data_dir, args.dilution)
# test_loader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = test_data.collate_fn, shuffle = True)

model = GNN(args.atom_num, args.emb_dim, args.hid_dims, args.fc_dims, args.dropout, test_data.labels)
model.load_checkpoint(os.path.join(args.model_dir, args.model_name))

tester = Tester(model, test_loader, test_data.labels, args.model_dir)
tester.predict(args.out_name)