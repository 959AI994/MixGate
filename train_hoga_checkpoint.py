from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import mixgate
import torch.distributed as dist
from config import get_parse_args
import mixgate.top_model
import mixgate.top_trainer

DATA_DIR = './data'

# --------------------------------------------
# CheckPoint
CKPT_PATH = '/home/xqgrp/wangjingxin/pythonproject/MixGate_aig/exp/01_hoga_0.00/01_hoga_0.00/model_last.pth'
# --------------------------------------------

if __name__ == '__main__':
    args = get_parse_args()
    circuit_path = '/home/xqgrp/wangjingxin/datasets/mixgate_data/merged_all1500.npz'
    num_epochs = args.num_epochs

    # ---------- 数据 ----------
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()

    # ---------- 模型 & Trainer ----------
    print('[INFO] Create Model and Trainer')
    model = mixgate.top_model.TopModel(
        args,
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )

    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed=True)
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1.0, 0.0, 1.0])

    # ---------- 尝试加载 checkpoint ----------
    start_epoch = 0
    if os.path.isfile(CKPT_PATH):
        trainer.load(CKPT_PATH)          # TopTrainer.load 会把 trainer.model_epoch 设置好
        start_epoch = trainer.model_epoch
        print(f'[INFO] Loaded checkpoint: {CKPT_PATH} (next epoch = {start_epoch})')
    else:
        print(f'[WARN] Checkpoint not found: {CKPT_PATH} — training from scratch')

    # ---------- 计算还需训练多少 ----------
    remain_epoch = num_epochs - start_epoch
    if remain_epoch <= 0:
        print('[INFO] num_epochs 已完成，无需继续训练')
        exit(0)

    # ---------- 开始 / 继续训练 ----------
    print('[INFO] Stage 1 Training ...')
    trainer.train(remain_epoch, train_dataset, val_dataset)
