#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-10-19 17:27:13
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')  

class DataPreprocess(object):
    """docstring for DataPreprocess"""
    def __init__(self, data_dir, fp):
        super(DataPreprocess, self).__init__()
        self.data_dir = data_dir
        self.fp = fp

    def process(self):
        self._read_smiles_descriptors()
        self._read_fps()
        self._read_labels()

        train_xs, train_ys = self._gen_x_y(self.train_df)
        val_xs, val_ys = self._gen_x_y(self.val_df)
        test_xs, test_ys = self._gen_x_y(self.test_df)
        return train_xs, train_ys, val_xs, val_ys, test_xs, test_ys

    def _read_smiles_descriptors(self):
        self.train_df = pd.read_excel(os.path.join(self.data_dir, "pyrfume_train.xlsx"))
        self.val_df = pd.read_excel(os.path.join(self.data_dir, "pyrfume_val.xlsx"))
        self.test_df = pd.read_excel(os.path.join(self.data_dir, "pyrfume_test.xlsx"))

    def _read_fps(self):
        self.fps = pickle.load(open(os.path.join(self.data_dir, "{}.pkl".format(self.fp)), "rb"))

    def _read_embs(self, data_name):
        embs = pd.read_csv(os.path.join(self.data_dir, data_name))
        columns = list(embs.columns)
        for c in columns:
            if c[:3] != "emb" and c != "smiles":
                embs.pop(c)
        smiles = embs.pop("smiles")
        fps = embs.values.squeeze()
        fps_dict = {s:fp for s, fp in zip(smiles, fps)}
        return fps_dict



    def _read_labels(self):
        self.label_list = pd.read_excel(os.path.join(self.data_dir, "labels.xlsx"))["labels"].values.squeeze().tolist()

    def _gen_x_y(self, df):
        xs, ys = [], []
        smiles, odors = df["SMILES"].values.squeeze().tolist(), df["Odor"].values.squeeze().tolist()
        for s, o in zip(smiles, odors):
            if s not in self.fps:
                continue
            fp = np.array(self.fps[s], dtype = np.float32)
            nanmask = np.isnan(fp)
            fp[nanmask] = 0

            descriptors = self._extract_odor(o)
            label = self._gen_label(descriptors)
            xs.append(fp)
            ys.append(label)
        xs, ys = np.array(xs), np.array(ys)
        print(xs.shape, ys.shape)
        return xs, ys

    def _gen_label(self, descriptors):
        label = np.zeros((len(self.label_list), ))
        for des in descriptors:
            idx = self.label_list.index(des)
            label[idx] = 1
        return label

    def _extract_odor(self, description):
        labels = [t.strip() for t in description.strip().replace("[", "").replace("]", "").replace("'", "").split(",")]
        return labels    

class SKClassifier(object):
    """docstring for ClassName"""
    def __init__(self, train, val, test, method, label_list, out_dir, fp):
        super(SKClassifier, self).__init__()
        self.train = train
        self.val = val
        self.test = test
        self.method = method
        self.label_list = label_list
        self.out_dir = os.path.join(out_dir, "{}_{}".format(fp, method))
        self.fp = fp
        os.makedirs(self.out_dir, exist_ok = True)

    def _cal_auc_auprc_metrics(self, preds, labels):
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds = {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        for i, l in enumerate(self.label_list):
            one_labels = labels[:, i]
            one_preds = preds[:, i]
            if sum(one_labels) == 0:
                continue
            auc = metrics.roc_auc_score(one_labels, one_preds)
            auprc = metrics.average_precision_score(one_labels, one_preds)

            threshold = self._cal_youden(one_preds, one_labels)
            precision, recall, specificity, f1, acc = self._cal_metrics((one_preds >= threshold).astype(int), one_labels)

            aucs[l] = auc
            nums[l] = sum(one_labels)/len(one_labels)
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

    def _cal_metrics(self, preds, labels):
        precision = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        specificity = metrics.recall_score(labels, preds, pos_label = 0)
        f1 = metrics.f1_score(labels, preds)
        acc = metrics.balanced_accuracy_score(labels, preds)
        return precision, recall, specificity, f1, acc

    def _draw_performances(self, nums, metrics, name, mark):
        ## add matplotlib image
        fig, ax = plt.subplots()
        for l in self.label_list:
            if l in metrics:
                ax.scatter(nums[l], metrics[l], label = l)
        ax.set_xlabel("# positive sample (%)")
        ax.set_ylabel("{} ({})".format(name, mark))
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.out_dir, "{}_{}.png".format(name, mark)), bbox_inches = "tight", dpi = 300)

    def _fit_clf_multilabel(self, clf):
        clf.fit(self.train[0], self.train[1])
        val_preds = clf.predict(self.val[0])
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds = self._cal_auc_auprc_metrics(val_preds, self.val[1])
        avg_auc = np.mean(list(aucs.values()))
        print(list(aucs.values()))
        return avg_auc, clf

    def _fit_clf_binary(self, clf, i):
        clf.fit(self.train[0], self.train[1][:, i])
        if sum(self.val[1][:, i]) != 0:
            val_preds = clf.predict(self.val[0])
            auc = metrics.roc_auc_score(self.val[1][:, i], val_preds)
            return auc, clf
        else:
            train_preds = clf.predict(self.train[0])
            auc = metrics.roc_auc_score(self.train[1][:, i], train_preds)
            return auc, clf        

    def run(self):
        if self.method == "smote-svm":
            self.best_clfs, self.best_aucs, self.best_params = {}, {}, {}
            for l in self.label_list:
                self.best_aucs[l] = -1
            # for k in ["linear", "poly", "rbf", "sigmoid", "precomputed"]:
            for k in ["linear"]:
                for i, l in tqdm(enumerate(self.label_list), ncols = 80):
                    m = SVC(kernel = k, random_state = 0, verbose = False)
                    resample = SMOTE()
                    #Define pipeline
                    clf = Pipeline(steps=[('r', resample), ('m', m)])

                    auc, clf = self._fit_clf_binary(clf, i)
                    if auc > self.best_aucs[l]:
                        self.best_clfs[l] = clf
                        self.best_aucs[l] = auc
                        self.best_params[l] = k
                        print(k, self.best_aucs[l])
        elif self.method in ["gb", "smote-gb", "knn", "smote-knn", "rf", "smote-rf"]:
            # print(self.method)
            self.best_clfs, self.best_aucs, self.best_params = {}, {}, {}
            for l in self.label_list:
                self.best_aucs[l] = -1
            # for param in tqdm(range(1, 201, 2), ncols = 80):
            for param in tqdm([10, 20, 50, 100, 200], ncols = 80):
                for i, l in enumerate(self.label_list):
                    if self.method == "gb":
                        clf = GradientBoostingClassifier(n_estimators = param, random_state = 0, verbose = 0)
                    elif self.method == "smote-gb":
                        m = GradientBoostingClassifier(n_estimators = param, random_state = 0, verbose = 0)
                        resample = SMOTE()
                        #Define pipeline
                        clf = Pipeline(steps=[('r', resample), ('m', m)])                        
                    elif self.method == "knn":
                        clf = KNeighborsClassifier(n_neighbors = param, n_jobs = 16)
                    elif self.method == "smote-knn":
                        m = KNeighborsClassifier(n_neighbors = param, n_jobs = 16)
                        resample = SMOTE()
                        #Define pipeline
                        clf = Pipeline(steps=[('r', resample), ('m', m)])
                    elif self.method == "rf":
                        clf = RandomForestClassifier(random_state = 0, min_samples_leaf = param, verbose = 0, n_jobs = -1)
                    elif self.method == "smote-rf":
                        m = RandomForestClassifier(random_state = 0, min_samples_leaf = param, verbose = 0, n_jobs = -1)
                        resample = SMOTE()
                        #Define pipeline
                        clf = Pipeline(steps=[('r', resample), ('m', m)])

                    auc, clf = self._fit_clf_binary(clf, i)
                    if auc > self.best_aucs[l]:
                        self.best_clfs[l] = clf
                        self.best_aucs[l] = auc
                        self.best_params[l] = param
                        print(i, param, self.best_aucs[l])

    def draw_preds(self, preds, labels, mark):
        xs = np.arange(len(preds))
        out_dir = os.path.join(self.out_dir, mark)
        os.makedirs(out_dir, exist_ok = True)
        for i, l in enumerate(self.label_list):
            df = pd.DataFrame()
            df["no"] = xs
            df["pred"] = preds[:, i]
            df["label"] = labels[:, i]
            fig = plt.figure()
            sns.scatterplot(data = df, x = "no", y = "pred", hue = "label")
            plt.savefig(os.path.join(out_dir, "{}.png".format(l)), bbox_inches = "tight")

    def evaluate(self, mark):
        if mark == "test":
            data = self.test
        elif mark == "val":
            data = self.val

        # if self.method in ["rf", "knn"]:
        #     best_param = self.best_param
        #     test_preds = self.best_clf.predict(data[0])
        
        if self.method in ["svm", "gb", "smote-gb", "knn", "smote-knn", "rf", "smote-rf"]:
            best_param = self.best_params
            test_preds = []
            for i, l in tqdm(enumerate(self.label_list), ncols = 80):
                test_pred = self.best_clfs[l].predict(data[0])
                test_preds.append(test_pred)
            test_preds = np.array(test_preds).transpose()
            print(test_preds.shape)
        self.draw_preds(test_preds, data[1], mark)
        aucs, auprcs, precisions, recalls, specificities, f1s, accs, nums, thresholds = self._cal_auc_auprc_metrics(test_preds, data[1])
   
        self._draw_performances(nums, aucs, "auc", mark)
        self._draw_performances(nums, auprcs, "auprc", mark)
        self._draw_performances(nums, precisions, "precision", mark)
        self._draw_performances(nums, f1s, "f1", mark)
        auc = np.mean(list(aucs.values()))
        auprc = np.mean(list(auprcs.values()))
        precision = np.mean(list(precisions.values()))
        recall = np.mean(list(recalls.values()))
        specificity = np.mean(list(specificities.values()))
        f1 = np.mean(list(f1s.values()))
        acc = np.mean(list(accs.values()))

        return auc, auprc, precision, recall, specificity, f1, acc, best_param

        
if __name__ == '__main__':
    dp = DataPreprocess("../../pyrfume_models3_sois/dumped30_coulomb_frobenius/", "mordreds")
    dp.process()