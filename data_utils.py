#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-14 15:41:57
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, requests, pickle, json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
# from googletrans import Translator
from matplotlib import pyplot as plt

from preprocess import Preprocessor

def download_smiles(cid):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/json".format(cid)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    }
    try:
        r = requests.get(
                url, 
                headers = headers)
        data = r.json()
        refs = data["Record"]["Section"]
        for ref in refs:
            if ref["TOCHeading"] == "Names and Identifiers":
                sub_refs = ref["Section"]
                for sub_ref in sub_refs:
                    if sub_ref["TOCHeading"] == "Computed Descriptors":
                        subsub_refs = sub_ref["Section"]
                        for subsub_ref in subsub_refs:
                            if subsub_ref["TOCHeading"] == "Canonical SMILES":
                                subsubsub_ref = subsub_ref["Information"][0]
                                smiles = subsubsub_ref["Value"]["StringWithMarkup"][0]["String"]
                                break
        return smiles, "SUCCESS"
    except Exception as e:
        return _, "FAIL"

def download_smiles_list(cids, save_path):
    f = open(save_path, "a+")
    for cid in cids:
        smiles, status = download_smiles(cid)
        if status == "SUCCESS":
            f.write("{},{}\n".format(cid, smiles))
            f.flush()
        else:
            print(cid)
    f.close()

def check_overlap_in_aicrowd(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "round-3-supplementary-training-data.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_smiles, test_smiles = train_df["SMILES"].values.squeeze().tolist(), test_df["SMILES"].values.squeeze().tolist()
    for test in test_smiles:
        if test in train_smiles:
            print(test)

def merge_voc_in_aicrowd(data_dir):
    test_path = os.path.join(data_dir, "round-3_vocabulary.txt")
    train_path = os.path.join(data_dir, "vocabulary.txt")
    train_vocs = pd.read_csv(train_path, header = None).values.squeeze().tolist()
    test_vocs = pd.read_csv(test_path, header = None).values.squeeze().tolist()

    for test in test_vocs:
        if test not in train_vocs:
            train_vocs.append(test)
            print(test)
    out_df = pd.DataFrame(train_vocs, columns = ["label"])
    out_df.sort_values(by = "label", inplace = True)
    out_df.to_csv(os.path.join(data_dir, "tot_vocabulary.csv"), index = False)
    
def split_dataset(data_path, out_dir, name, train_ratio = 0.8):
    df = pd.read_excel(data_path)
    data = df.values
    data_num = len(data)
    train_num = int(data_num*train_ratio)
    train_idxs = np.random.choice(data_num, train_num, replace = False)
    test_idxs = np.array([t for t in range(data_num) if t not in train_idxs])
    train_df = pd.DataFrame(data[train_idxs], columns = df.columns)
    test_df = pd.DataFrame(data[test_idxs], columns = df.columns)
    os.makedirs(out_dir, exist_ok = True)
    print(os.path.join(out_dir, "{}_train.xlsx".format(name)))
    train_df.to_excel(os.path.join(out_dir, "{}_train.xlsx".format(name)), index = False)
    test_df.to_excel(os.path.join(out_dir, "{}_test.xlsx".format(name)), index = False)

def split_dataset_train_val_test(data_path, out_dir, name, train_ratio = 0.8):
    df = pd.read_excel(data_path)
    data = df.values
    data_num = len(data)
    train_num = int(data_num*train_ratio)
    train_idxs = np.random.choice(data_num, train_num, replace = False)
    val_test_idxs = np.array([t for t in range(data_num) if t not in train_idxs])
    val_idxs = np.random.choice(val_test_idxs, len(val_test_idxs)//2, replace = False)
    test_idxs = np.array([t for t in val_test_idxs if t not in val_idxs])

    train_df = pd.DataFrame(data[train_idxs], columns = df.columns)
    val_df = pd.DataFrame(data[val_idxs], columns = df.columns)
    test_df = pd.DataFrame(data[test_idxs], columns = df.columns)
    os.makedirs(out_dir, exist_ok = True)
    train_df.to_excel(os.path.join(out_dir, "{}_train.xlsx".format(name)), index = False)
    val_df.to_excel(os.path.join(out_dir, "{}_val.xlsx".format(name)), index = False)
    test_df.to_excel(os.path.join(out_dir, "{}_test.xlsx".format(name)), index = False)
    print(train_df.shape, val_df.shape, test_df.shape)

def split_dataset_train_val_test_SOIS(data_path, label_path, out_dir, name, train_ratio = 0.8):
    from skmultilearn.model_selection import IterativeStratification

    df = pd.read_excel(data_path)
    data = df.values
    df["Odor"] = df["Odor"].str.replace("[", "")
    df["Odor"] = df["Odor"].str.replace("]", "")
    df["Odor"] = df["Odor"].str.replace("'", "")
    df["Odor"] = df["Odor"].str.replace(" ", "")
    odors = df["Odor"].values.squeeze()
    X = df.values.squeeze()

    ## odor to onehot
    labels = pd.read_excel(label_path)["labels"].values.squeeze()
    label2id = {label:i for i,label in enumerate(labels)}
    id2label = {i:label for i,label in enumerate(labels)}
    onehots = []
    for odor in odors:
        odor_list = odor.split(",")
        onehot = [1 if label in odor_list else 0 for label in label2id.keys()]
        onehots.append(onehot)
    Y = np.array(onehots)
    draw_pos_ratios(Y, id2label, out_dir, "total")

    kfolds = IterativeStratification(n_splits = 10, order = 2)
    for train_val, test in kfolds.split(X, Y):
        train_val_df = pd.DataFrame(X[train_val], columns = df.columns)
        test_df = pd.DataFrame(X[test], columns = df.columns)
        train_val_Y = Y[train_val]
        test_Y = Y[test]
        break

    train_val_X = train_val_df.values.squeeze()
    kfolds = IterativeStratification(n_splits = 9, order = 2)
    for train, val in kfolds.split(train_val_X, train_val_Y):
        train_df = pd.DataFrame(train_val_X[train], columns = df.columns)
        val_df = pd.DataFrame(train_val_X[val], columns = df.columns)
        train_Y = train_val_Y[train]
        val_Y = train_val_Y[val]
        break   

    draw_pos_ratios(train_Y, id2label, out_dir, "train")
    draw_pos_ratios(val_Y, id2label, out_dir, "val")
    draw_pos_ratios(test_Y, id2label, out_dir, "test")

    os.makedirs(out_dir, exist_ok = True)
    train_df.to_excel(os.path.join(out_dir, "{}_train_sois.xlsx".format(name)), index = False)
    val_df.to_excel(os.path.join(out_dir, "{}_val_sois.xlsx".format(name)), index = False)
    test_df.to_excel(os.path.join(out_dir, "{}_test_sois.xlsx".format(name)), index = False)
    print(train_df.shape, val_df.shape, test_df.shape)

def draw_pos_ratios(Y, id2label, out_dir, mark):
    num = len(Y)
    ## statistics of labels
    label2num = {}
    for k, v in id2label.items():
        pos = sum(Y[:, k])
        label2num[v] = pos
    draw_pos_ratio(label2num, out_dir, mark, "label")

    ## statistics of labelsets
    labelset2num = {}
    for y in Y:
        odor = [id2label[i] for i, id_ in enumerate(y) if id_ == 1]
        odor_str = ";".join(odor)
        if odor_str in labelset2num:
            labelset2num[odor_str] += 1
        else:
            labelset2num[odor_str] = 1
    draw_pos_ratio(labelset2num, out_dir, mark, "labelset")

def draw_pos_ratio(mapper, out_dir, mark, out_name, log = False):
    sorted_mapper = sorted(mapper.items(), key = lambda kv: (kv[1], kv[0]))

    ## add matplotlib image
    fig, ax = plt.subplots()
    cnter = 0
    for k, v in sorted_mapper:
        if log:
            v = np.log(v)
        ax.bar(cnter, v, label = k)
        cnter += 1
    print(k, v)
    ax.set_xlabel(out_name)
    ax.set_ylabel("pos ratio")
    # ax.set_ylim(0, 1)
    ax.set_xticks([])
    plt.savefig(os.path.join(out_dir, "{}_{}.png".format(mark, out_name)), bbox_inches = "tight", dpi = 300)

def extract_labels(data_path, out_dir, name):
    df = pd.read_excel(data_path)
    labels_list = df["Odor"].values.squeeze()
    labels = []
    for l in labels_list:
        l = l.strip().replace("[", "").replace("]", "").replace("'", "").split(",")
        # if len(l) > 1:
        #     print(l)
        for t in l:
            t = t.strip()
            if t not in labels:
                labels.append(t)
    labels = sorted(labels)
    out_df = pd.DataFrame(labels, columns = ["labels"])
    out_df.to_excel(os.path.join(out_dir, "labels_{}.xlsx".format(name)), index= False)

def gen_pyrfume_cmds(split):
    cnt = 0
    for i in range(split):
        with open("train_pyrfume{}.bat".format(i), "a+") as f:
            line = "cd C:\\Users\\mengji.DESKTOP-U4SLS3J\\Desktop\\mengji_codes\\scentAI\\codes\n"
            f.write(line)         
            line = "CALL conda.bat activate rdkit\n"
            f.write(line)

    # out_path = "../models_GS_addHs_laplace_cat5/results.csv"
    # res = pd.read_csv(out_path, header = None)
    # res.columns = ["model", "lr", "drop", "emb", "hid", "lap", "nlayer", "freq", "val_loss", "val_auc", "dilution", "r"]

    for hid_layer in [4, 6, 8]:
        for lr in [1, 0.1]:
            # for cou in ["zscore", "none", "frobenius", "minmax", "bin20", "adj"]:
            for cou in ["none"]:
                        # # check whether trained or not
                        # idx = (res["lap"] == lap_dim) & (res["freq"] == max_freq) & (res["lr"] == lr) & (res["drop"] == drop) & (res["emb"] == emb) & (res["hid"] == hid_dim) & (res["nlayer"] == hid_layer)
                        # if len(res[idx]) == 3:
                        #     print(res[idx].shape)
                        #     continue

                if cou == "adj":
                    matrix = "adjacent"
                else:
                    matrix = "coulomb"
                if cou == "bin20":
                    step = 20
                else:
                    step = -1
                cnt += 1
                device = cnt % 4
                file = cnt % split   
                with open("train_pyrfume{}.bat".format(file), "a+") as f:
                    line = "python train_pyrfume.py --coulomb_bin_step {} --gnn_matrix {} --out_dir ../dream_pyrfume_models2/GNN_addHs5_dumped30_cou{}/ --data_dir ../dream_pyrfume_models2/dumped30_coulomb_{}/ --lap_dim {} --max_freq {} --lr {} --device {} --dropout {} --emb_dim {} --hid_dim {} --hid_layer {}\n".format(
                        step, matrix, cou, cou, 16, 16, lr, device, 0.1, 32, 16, hid_layer)
                    f.write(line)  

def gen_GS_cmds_coulomb(split):
    cnt = 0
    for i in range(split):
        with open("train_GS_coulomb_minmax{}.bat".format(i), "a+") as f:
            line = "cd C:\\Users\\mengji.DESKTOP-U4SLS3J\\Desktop\\mengji_codes\\scentAI\\codes\n"
            f.write(line)         
            line = "CALL conda.bat activate rdkit\n"
            f.write(line)

    # out_path = "../models_GS_addHs_coulomb_frobenius2/results.csv"
    # res = pd.read_csv(out_path, header = None)
    # res.columns = ["model", "lr", "drop", "emb", "hid", "lap", "nlayer", "freq", "val_loss", "val_auc", "dilution", "r"]
    for drop in [0.1]:
        for emb in [32]:
            for hid_dim in [16, 32]:
                for hid_layer in [2, 4, 6, 8]:
                    for lr in [1, 0.1, 0.01, 0.001]:                     
                        # # check whether trained or not
                        # idx = (res["lr"] == lr) & (res["drop"] == drop) & (res["emb"] == emb) & (res["hid"] == hid_dim) & (res["nlayer"] == hid_layer)
                        # if len(res[idx]) == 3:
                        #     print(res[idx].shape)
                        #     continue
                        cnt += 1
                        device = cnt % 4
                        file = cnt % split   
                        with open("train_GS_coulomb_minmax{}.bat".format(file), "a+") as f:
                            line = "python train_goodscents_coulomb3.py --lap_dim {} --max_freq {} --lr {} --device {} --dropout {} --emb_dim {} --hid_dim {} --hid_layer {}\n".format(
                                16, 16, lr, device, round(drop, 2), emb, hid_dim, hid_layer)
                            f.write(line)  

def gen_GSLPE_cmds():
    split, cnt = 4, 0
    for i in range(split):
        with open("train_GS_LPE{}.bat".format(i), "a+") as f:
            line = "cd C:\\Users\\mengji.DESKTOP-U4SLS3J\\Desktop\\mengji_codes\\scentAI\\codes\n"
            f.write(line)         
            line = "CALL conda.bat activate rdkit\n"
            f.write(line)

    out_path = "../models_GS_addHs_lpe/results.csv"
    res = pd.read_csv(out_path, header = None)
    res.columns = ["model", "lr", "drop", "nheads", "lpe_dim", "lpe_layer", "hid_dim", "nlayer", "freq", "val_loss", "val_auc", "dilution", "r"]
    for drop in [0.1]:
        for hid_dim in [16]:
            for hid_layer in [2, 4, 6, 8]:
                for lr in [1, 0.1, 0.01, 0.001]:
                    for max_freq in [2, 5, 10, 20]:
                        for lpe_dim in [8, 16]:
                            for nhead in [4, 8]:
                                for lpe_layer in [2, 4]:
                                    # check whether trained or not
                                    idx = (res["lpe_layer"] == lpe_layer) & (res["lpe_dim"] == lpe_dim) & (res["freq"] == max_freq) & (res["lr"] == lr) & (res["drop"] == drop) & (res["nheads"] == nhead) & (res["hid_dim"] == hid_dim) & (res["nlayer"] == hid_layer)
                                    if len(res[idx]) == 3:
                                        print(res[idx].shape, idx)
                                        continue
                                    cnt += 1
                                    device = cnt % 4
                                    file = cnt % split   
                                    with open("train_GS_LPE{}.bat".format(file), "a+") as f:
                                        line = "python train_goodscents_lpe.py --LPE_layer {} --LPE_n_heads {} --LPE_dim {} --max_freq {} --lr {} --device {} --dropout {} --hid_dim {} --hid_layer {}\n".format(
                                            lpe_layer, nhead, lpe_dim, max_freq, lr, device, round(drop, 2), hid_dim, hid_layer)
                                        f.write(line)  

def merge_labels_mannully(label_path, out_dir, start_idx, end_idx):
    translator = Translator(service_urls = ["translate.google.cn"])

    labels = pd.read_excel(label_path)["labels"].values.squeeze()
    raw2label, label2raw, last_key, i = {}, {}, labels[0], start_idx
    while i < end_idx:
        print("=="*20)
        if i != 0:
            print("LAST: {} {}\nTHIS: {} {}".format(
                labels[i-1], 
                translator.translate(labels[i-1], dest = "zh-CN").text, 
                labels[i],
                translator.translate(labels[i], dest = "zh-CN").text, 
                ))
        else:
            print("FIRST: {}".format(labels[i]))
        new_key = input("1 for the same, 2 for the different")
        if new_key == "2":
            last_key = labels[i]
            raw2label[labels[i]] = last_key
            label2raw[last_key] = [labels[i]]
            i += 1
        elif new_key == "1":
            raw2label[labels[i]] = last_key
            label2raw[last_key].append(labels[i])
            i += 1
        # elif new_key == "3":
        #     label2raw[last_key].pop()
        #     del raw2label[labels[i]]
        #     i -= 1

    with open(os.path.join(out_dir, "raw2label{}_{}.json".format(start_idx, end_idx)), "w") as f:
        json.dump(raw2label, f)
    with open(os.path.join(out_dir, "label2raw{}_{}.json".format(start_idx, end_idx)), "w") as f:
        json.dump(label2raw, f)

if __name__ == '__main__':
    # train_path = "../data_DREAM/TrainSet.txt"
    # out_path = "../data_DREAM/cid2smiles.txt"
    # cids = sorted(set(pd.read_csv(train_path, sep = "\t", header = 0)["Compound Identifier"].values.squeeze().tolist()))
    # download_smiles_list(cids, out_path)

    # train_path = "../data_DREAM/CID_testset.txt"
    # out_path = "../data_DREAM/cid2smiles.txt"
    # cids = sorted(set(pd.read_csv(train_path, sep = "\t", header = None).values.squeeze().tolist()))
    # download_smiles_list(cids, out_path)

    # split, cnt = 8, 0
    # label_path = "../data_DREAM/label.txt"
    # labels = pd.read_csv(label_path)["labels"].values.squeeze().tolist()
    # for dilu in ["10", "100000"]:
    #     for model in ["GNN", "SimpleNN"]:
    #         for drop in np.arange(0, 1, 0.05):
    #             for l in labels:
    #                 cnt += 1
    #                 device = cnt % 4
    #                 file = cnt % 8 
    #                 with open("run_simple{}.bat".format(file), "a+") as f:
    #                     line = "python train.py --dilution {} --device {} --model {} --dropout {} --wanted_label {}\n".format(dilu, device, model, round(drop, 2), l)
    #                     f.write(line)

    # data_dir = "../data_AIcrowd/"
    # # check_overlap_in_aicrowd(data_dir)
    # merge_voc_in_aicrowd(data_dir)  

    # data_path = "../crawled_data/goodscents_odor.xlsx"
    # # split_dataset(data_path, "../data_GoodScents")
    # extract_labels(data_path, "../data_GoodScents")

    # gen_GS_cmds(8)
    # gen_GSLPE_cmds()
    # gen_GS_cmds_coulomb(4)

    # data_path = "../data_pyrfume/GdLef1000_total.xlsx"
    # out_dir = "../data_pyrfume/"
    # name = "GdLef1000"
    # split_dataset(data_path, out_dir, name)
    # extract_labels(data_path, out_dir, name)

    # merge_labels_mannully("../data_pyrfume/labels.xlsx", "../data_pyrfume", 0, 597)
    # with open(os.path.join(out_dir, "label2raw.json"), "r") as f:
    #     label2raw = json.load(f)
    # print(len(label2raw))

    # gen_pyrfume_cmds(6)

    # data_path = "../data_pyrfume/ArcGdLef30_total.xlsx"
    # out_dir = "../data_pyrfume/"
    # name = "ArcGdLef30"
    # split_dataset(data_path, out_dir, name)
    # extract_labels(data_path, out_dir, name)

    # data_path = "../data_pyrfume/10db30_total.xlsx"
    # out_dir = "../data_pyrfume/"
    # name = "10db30"
    # split_dataset_train_val_test(data_path, out_dir, name)
    # extract_labels(data_path, out_dir, name)

    data_path = "../data_pyrfume/10db30_total.xlsx"
    label_path = "../data_pyrfume/10db30_labels.xlsx"
    out_dir = "../data_pyrfume/"
    name = "10db30"
    split_dataset_train_val_test_SOIS(data_path, label_path, out_dir, name)