from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args
import mixgate.top_model
import mixgate.top_trainer 
import torch.distributed as dist

DATA_DIR = './data'

if __name__ == '__main__':
    args = get_parse_args()

    # ---------------------- 基础配置（与你原始脚本一致） ----------------------
    circuit_path = '/home/lcm/data/merged_mixgate1%1500.npz'  # 你的数据集路径
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()

    print('[INFO] Create Model and Trainer')
    model = mixgate.top_model.TopModel(
        args, 
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )
    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed=True)

    # ---------------------- 关键修改：从阶段二权重开始训练 ----------------------
    # 加载阶段二的完整模型权重（你之前训练好的 stage2_full_model.pth）
    stage2_ckpt_path = '/home/lcm/wangjingxin/pythonproject/MixGate_aig/exp/01_deepgate2_0.03_alignment/01_deepgate2_0.03_alignment/stage2_full_model.pth'
    print(f'[INFO] Loading Stage 2 checkpoint from: {stage2_ckpt_path}')
    trainer.load(stage2_ckpt_path)  # 直接加载阶段二的权重

    # ---------------------- 冻结GNN encoder参数（关键操作） ----------------------
    print('[INFO] Freezing GNN encoder parameters...')
    # # 模型中GNN encoder的参数名包含 "gnn_encoder"（需根据实际模型结构调整）
    # for name, param in model.named_parameters():
    #     if 'gnn_encoder' in name:  # 关键：匹配GNN encoder的参数名
    #         param.requires_grad = False  # 冻结参数，不参与训练

    # ---------------------- 配置后续训练的超参数 ----------------------
    # 新的学习率（建议比阶段二小，例如1e-5，可根据实际效果调整）
    new_lr = 1e-4  
    # 总训练epoch：120（前60仅prob，后60 prob+func）
    total_epochs_after_stage2 = 120
    # ----------------------

    # ---------------------- 阶段三：前60 epoch 仅训练prob ----------------------
    print('[INFO] Start Stage 3: Training prob only (60 epochs)...')
    # 设置训练参数：仅prob损失生效（loss_weight顺序：[prob, mcm, func, align]）
    trainer.set_training_args(
        lr=new_lr,          # 新学习率
        lr_step=50,         # 学习率衰减步数（根据你的总epoch调整）
        loss_weight=[1.0, 0.0, 0.0, 0.0]  # 仅prob权重为1，其他为0
    )
    # 训练60个epoch
    trainer.train(60, train_dataset, val_dataset)
    # 保存阶段三的权重（可选，但建议保留）
    stage3_ckpt_path = os.path.join(trainer.log_dir, 'stage3_prob_only_model.pth')
    trainer.save(stage3_ckpt_path)
    print(f'[INFO] Stage 3 completed. Model saved to: {stage3_ckpt_path}')

    # ---------------------- 阶段四：后60 epoch 训练prob+func ----------------------
    print('[INFO] Start Stage 4: Training prob and func (60 epochs)...')
    # 设置训练参数：prob和func损失生效
    trainer.set_training_args(
        lr=new_lr,          # 保持学习率（或根据需要调整）
        lr_step=50,         
        loss_weight=[1.0, 0.0, 1.0, 0.0]  # prob和func权重为1，其他为0
    )
    # 继续训练60个epoch（总epoch累计120）
    trainer.train(60, train_dataset, val_dataset)
    # 保存最终模型
    final_ckpt_path = os.path.join(trainer.log_dir, 'final_full_model.pth')
    trainer.save(final_ckpt_path)
    print(f'[INFO] All stages completed. Final model saved to: {final_ckpt_path}')