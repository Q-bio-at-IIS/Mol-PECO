#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-09-01 13:02:47
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
from models import GNN

def load_pretrained_infograph(gnn, pretrained_path):
    gnn_dict = gnn.state_dict()

    pretrained_sd = torch.load(pretrained_path)
    pretrained_dict = {k: v for k, v in pretrained_sd.items() if "encoder" in k}
    gnn_dict.update(pretrained_dict)
    gnn.load_state_dict(gnn_dict)
    print("load model from {}".format(pretrained_path))

    # for k, v in gnn.named_parameters():
    #     if "encoder" in k:
    #         v.requires_grad = False
    return gnn



if __name__ == '__main__':
    pretrained_path = "../model_pretrain_by_infograph/zinc2m_aicrowd_dream/lr0.001_regu1.0/min.ckpt"
    load_pretrained_infograph(pretrained_path)