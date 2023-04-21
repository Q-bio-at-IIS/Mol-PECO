#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-08-17 19:41:26
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt


# test with random forest
def train_val_reg(train_xs, train_ys, val_xs, val_ys, min_samples_leaf = 1):
    reg = RandomForestRegressor(random_state = 0, min_samples_leaf = min_samples_leaf)
    reg.fit(train_xs, train_ys)
    val_pred = reg.predict(val_xs)
    r, p = pearsonr(val_pred, val_ys)
    return r, reg

def scatter_plot(preds, labels, r, out_name):
    fig = plt.figure(figsize = (3,3), dpi = 300)
    plt.scatter(labels, preds, alpha = 0.8, color = "red")
    m = LinearRegression(fit_intercept = False)
    m.fit(labels.reshape(-1, 1), preds.reshape(-1, 1))
    xs = np.arange(-0.05, 1.10, 0.05).reshape(-1, 1)
    pred_b = m.predict(xs)
    plt.plot(xs, pred_b, color = "blue", lw = 2, alpha = 0.8)
    plt.xlabel("True")
    plt.ylabel("Pred")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.title("Pearson of {}".format(round(r, 3)))
    plt.savefig(out_name, bbox_inches = "tight")

def filter_zero(xs, ys):
    mask = ys == 0
    nonzero = ~mask
    return xs[nonzero], ys[nonzero]

def finetune_rf(df, wanted_label, out_dir):
    df2 =  df.copy()
    columns = list(df2.columns)
    for c in columns:
        if "emb" in c or c == wanted_label:
            continue
        else:
            df2.pop(c)
    ys = df2.pop(wanted_label).values.squeeze()
    xs = df2.values
    splits = df["split"].values.squeeze()
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"
    train_xs, val_xs, test_xs = xs[train_mask], xs[val_mask], xs[test_mask]
    train_ys, val_ys, test_ys = ys[train_mask], ys[val_mask], ys[test_mask]

    train_xs, train_ys = filter_zero(train_xs, train_ys)
    val_xs, val_ys = filter_zero(val_xs, val_ys)
    test_xs, test_ys = filter_zero(test_xs, test_ys)

    best_r, best_reg, best_leaf = -99, None, None
    for min_samples_leaf in range(1, 51):
        val_r, reg = train_val_reg(train_xs, train_ys, val_xs, val_ys, min_samples_leaf)
        if val_r > best_r:
            best_r = val_r
            best_reg = reg
            best_leaf = min_samples_leaf
    test_pred = best_reg.predict(test_xs)
    test_r, test_p = pearsonr(test_pred, test_ys)
    scatter_plot(test_pred, test_ys, test_r, os.path.join(out_dir, "{}.png".format(wanted_label.replace("/", ""))))
    print("{}: {} ({}), with best leaf of {}".format(wanted_label, test_r, test_p, best_leaf))
    return test_r, test_p
