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

    # circuit_path ='/home/lcm/data/merged_all.npz'
    # circuit_path ='/home/lcm/wangjingxin/datasets/mixgate_data/merged_all150.npz'
    # circuit_path ='/home/lcm/wangjingxin/datasets/mixgate_data/merged_mixgate_consistency.npz'
    # circuit_path ='/gz-data/dchacker_1/merge_all1%15000.npz'
    circuit_path ='/root/pythonproject/datasets/merged_mixgate1%1500.npz'
    # circuit_path ='/home/lcm/wangjingxin/datasets/mixgate_data/merged_mixgate1508_consistency.npz'
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)

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
    
    # 第一阶段：60个epoch，训练prob（不使用对齐损失）
    print('[INFO] Stage 1 Training: Prob training only (60 epochs) ...')
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1.0, 0.1, 0.0, 0.1])
    trainer.train(60, train_dataset, val_dataset)
    
    # 保存第一阶段训练结束后的权重
    trainer.save(os.path.join(trainer.log_dir, 'stage1_prob_only_model.pth'))
    
    # 加载第一阶段的模型权重
    print('[INFO] Loading Stage 1 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage1_prob_only_model.pth'))

    # 第二阶段：60个epoch，训练prob和func（不使用对齐损失）
    print('[INFO] Stage 2 Training: Prob and Func training (60 epochs) ...')
    trainer.set_training_args(lr=1e-3, lr_step=50, loss_weight=[1.0, 0.1, 1.0, 0.1])
    trainer.train(60, train_dataset, val_dataset)
    
    # 保存第二阶段训练结束后的权重
    trainer.save(os.path.join(trainer.log_dir, 'stage2_prob_func_model.pth'))
    
    # 加载第二阶段的模型权重
    print('[INFO] Loading Stage 2 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage2_prob_func_model.pth'))

    # 第三阶段：最后60个epoch，训练prob、func和mask（不使用对齐损失）
    print('[INFO] Stage 3 Training: Prob, Func, and Mask training (60 epochs) ...')
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1.0, 0.0, 1.0, 0.0])
    trainer.train(60, train_dataset, val_dataset)
    
    # 保存最终模型
    trainer.save(os.path.join(trainer.log_dir, 'stage3_final_model.pth'))
    
    print('[INFO] Training completed!')
    print('[INFO] Stage 1: Prob training only (60 epochs)')
    print('[INFO] Stage 2: Prob and Func training (60 epochs)')
    print('[INFO] Stage 3: Prob, Func, and Mask training (60 epochs)')
    