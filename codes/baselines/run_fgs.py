#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-19 17:06:46
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, argparse
import pandas as pd
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--fp", default = "mordreds", type = str) ##bfps, cfps, mordreds
# parser.add_argument("--data_dir", default = "../../pyrfume_models3_sois/GNNLPE_addHs7_dumped30_coufrobenius/0.0_0.25_0.75/lr0.01_dim16/pred_emb_afterNN/", type = str)
parser.add_argument("--data_dir", default = "../../pyrfume_sois_canon/dumped30_coulomb_frobenius/", type = str)
parser.add_argument("--out_dir", default = "../../pyrfume_sois_canon/baselines_clf_50params/", type = str)
parser.add_argument("--model", default = "smote-rf", type = str) ##rf, knn, svm, gb
args = parser.parse_args()

dp = DataPreprocess(args.data_dir, args.fp)
train_xs, train_ys, val_xs, val_ys, test_xs, test_ys = dp.process()
print(train_xs.shape)

classifier = SKClassifier(
    (train_xs, train_ys), 
    (val_xs, val_ys), 
    (test_xs, test_ys), 
    args.model, 
    dp.label_list, 
    args.out_dir, 
    args.fp)
classifier.run()

val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc, best_param = classifier.evaluate("val")
test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc, best_param = classifier.evaluate("test")

res_path = os.path.join(args.out_dir, "res.csv")
with open(res_path, "a+") as f:
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        args.fp, args.model, best_param, 
        val_auc, val_auprc, val_precision, val_recall, val_spe, val_f1, val_acc,
        test_auc, test_auprc, test_precision, test_recall, test_spe, test_f1, test_acc))