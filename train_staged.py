from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data'

def load_preprocessed_data(data_dir, dataset_name, batch_size=16, trainval_split=1.0):
    """直接使用现有的数据加载逻辑"""
    print(f'[INFO] Loading preprocessed data from {data_dir}/{dataset_name}')
    
    # 使用现有的NpzParser_Pair，传入一个虚拟的npz路径（不会被使用）
    dataset = mixgate.NpzParser_Pair(data_dir, 'dummy.npz')
    train_dataset, val_dataset = dataset.get_dataset()
    
    print(f'[INFO] Loaded {len(train_dataset) + len(val_dataset)} graphs, train: {len(train_dataset)}, val: {len(val_dataset)}')
    
    return train_dataset, val_dataset

def train_stage1_encoders(args, train_dataset, val_dataset):
    """
    第一阶段：训练4个GNN编码器（AIG, MIG, XMG, XAG）
    目标：对齐损失 + 概率预测损失 + TT损失
    """
    print('[INFO] ===== Stage 1: Training 4 GNN Encoders =====')
    
    # 训练AIG编码器
    print('[INFO] Training AIG Encoder...')
    if args.aig_encoder == 'dg2':
        aig_model = mixgate.dg_model.Model(dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'pg':
        aig_model = mixgate.aig_encoder.polargate.PolarGate(args, in_dim=3, out_dim=args.dim_hidden)
    elif args.aig_encoder == 'dg3':
        aig_model = mixgate.aig_encoder.deepgate3.DeepGate3(dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'gcn':
        aig_model = mixgate.aig_encoder.gcn.DirectMultiGCNEncoder(dim_feature=3, dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'hoga':
        aig_model = mixgate.aig_encoder.hoga.HOGA(in_channels=3, hidden_channels=args.dim_hidden, 
                                                 out_channels=args.dim_hidden, num_layers=1,
                                                 dropout=0.1, num_hops=5+1, heads=8, directed=True, attn_type="mix")
    
    # 修改args.save_dir指向统一目录
    original_save_dir = args.save_dir
    args.save_dir = os.path.join(args.save_dir, 'unified')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    aig_trainer = mixgate.dc_trainer.Trainer(args, aig_model, distributed=True, device='cuda:0')
    
    # AIG编码器双阶段训练
    print('[INFO] AIG Encoder Stage 1: Training probability prediction (60 epochs)...')
    aig_trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=50)  # prob_loss only
    aig_trainer.train(60, train_dataset, val_dataset)
    
    print('[INFO] AIG Encoder Stage 2: Training TT loss (60 epochs)...')
    aig_trainer.set_training_args(loss_weight=[0.0, 0.0, 1.0], lr=1e-4, lr_step=50)  # tt_loss only
    aig_trainer.train(60, train_dataset, val_dataset)
    
    aig_ckpt_path = os.path.join(args.save_dir, 'aig_encoder_stage1.pth')
    aig_trainer.save(aig_ckpt_path)
    
    # 复制日志文件到统一目录
    import shutil
    if hasattr(aig_trainer, 'log_path') and os.path.exists(aig_trainer.log_path):
        log_filename = os.path.basename(aig_trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'aig_{log_filename}')
        shutil.copy2(aig_trainer.log_path, unified_log_path)
        print(f'[INFO] AIG training log saved to: {unified_log_path}')
    
    # 销毁分布式进程组，为下一个trainer做准备
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    # 训练MIG编码器
    print('[INFO] Training MIG Encoder...')
    mig_model = mixgate.dg_model_mig.Model(dim_hidden=args.dim_hidden)
    mig_trainer = mixgate.dc_trainer.Trainer(args, mig_model, distributed=True, device='cuda:0')
    
    # MIG编码器双阶段训练
    print('[INFO] MIG Encoder Stage 1: Training probability prediction (60 epochs)...')
    mig_trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=50)  # prob_loss only
    mig_trainer.train(60, train_dataset, val_dataset)
    
    print('[INFO] MIG Encoder Stage 2: Training TT loss (60 epochs)...')
    mig_trainer.set_training_args(loss_weight=[0.0, 0.0, 1.0], lr=1e-4, lr_step=50)  # tt_loss only
    mig_trainer.train(60, train_dataset, val_dataset)
    
    mig_ckpt_path = os.path.join(args.save_dir, 'mig_encoder_stage1.pth')
    mig_trainer.save(mig_ckpt_path)
    
    # 复制日志文件到统一目录
    if hasattr(mig_trainer, 'log_path') and os.path.exists(mig_trainer.log_path):
        log_filename = os.path.basename(mig_trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'mig_{log_filename}')
        shutil.copy2(mig_trainer.log_path, unified_log_path)
        print(f'[INFO] MIG training log saved to: {unified_log_path}')
    
    # 销毁分布式进程组，为下一个trainer做准备
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    # 训练XMG编码器
    print('[INFO] Training XMG Encoder...')
    xmg_model = mixgate.dg_model_xmg.Model(dim_hidden=args.dim_hidden)
    xmg_trainer = mixgate.dc_trainer.Trainer(args, xmg_model, distributed=True, device='cuda:0')
    
    # XMG编码器双阶段训练
    print('[INFO] XMG Encoder Stage 1: Training probability prediction (60 epochs)...')
    xmg_trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=50)  # prob_loss only
    xmg_trainer.train(60, train_dataset, val_dataset)
    
    print('[INFO] XMG Encoder Stage 2: Training TT loss (60 epochs)...')
    xmg_trainer.set_training_args(loss_weight=[0.0, 0.0, 1.0], lr=1e-4, lr_step=50)  # tt_loss only
    xmg_trainer.train(60, train_dataset, val_dataset)
    
    xmg_ckpt_path = os.path.join(args.save_dir, 'xmg_encoder_stage1.pth')
    xmg_trainer.save(xmg_ckpt_path)
    
    # 复制日志文件到统一目录
    if hasattr(xmg_trainer, 'log_path') and os.path.exists(xmg_trainer.log_path):
        log_filename = os.path.basename(xmg_trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'xmg_{log_filename}')
        shutil.copy2(xmg_trainer.log_path, unified_log_path)
        print(f'[INFO] XMG training log saved to: {unified_log_path}')
    
    # 销毁分布式进程组，为下一个trainer做准备
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    # 训练XAG编码器
    print('[INFO] Training XAG Encoder...')
    xag_model = mixgate.dg_model_xag.Model(dim_hidden=args.dim_hidden)
    xag_trainer = mixgate.dc_trainer.Trainer(args, xag_model, distributed=True, device='cuda:0')
    
    # XAG编码器双阶段训练
    print('[INFO] XAG Encoder Stage 1: Training probability prediction (60 epochs)...')
    xag_trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=50)  # prob_loss only
    xag_trainer.train(60, train_dataset, val_dataset)
    
    print('[INFO] XAG Encoder Stage 2: Training TT loss (60 epochs)...')
    xag_trainer.set_training_args(loss_weight=[0.0, 0.0, 1.0], lr=1e-4, lr_step=50)  # tt_loss only
    xag_trainer.train(60, train_dataset, val_dataset)
    
    xag_ckpt_path = os.path.join(args.save_dir, 'xag_encoder_stage1.pth')
    xag_trainer.save(xag_ckpt_path)
    
    # 复制日志文件到统一目录
    if hasattr(xag_trainer, 'log_path') and os.path.exists(xag_trainer.log_path):
        log_filename = os.path.basename(xag_trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'xag_{log_filename}')
        shutil.copy2(xag_trainer.log_path, unified_log_path)
        print(f'[INFO] XAG training log saved to: {unified_log_path}')
    
    # 销毁分布式进程组，为下一个trainer做准备
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    return aig_ckpt_path, mig_ckpt_path, xmg_ckpt_path, xag_ckpt_path

def train_stage2_alignment(args, train_dataset, val_dataset, encoder_ckpts):
    """
    第二阶段：训练编码器对齐
    目标：对齐损失 + 概率预测损失
    """
    print('[INFO] ===== Stage 2: Training Encoder Alignment =====')
    
    aig_ckpt_path, mig_ckpt_path, xmg_ckpt_path, xag_ckpt_path = encoder_ckpts
    
    # 创建TopModel并加载预训练的编码器
    model = mixgate.TopModel(args, aig_ckpt_path, mig_ckpt_path, xmg_ckpt_path, xag_ckpt_path)
    
    # 冻结编码器参数
    for encoder in [model.deepgate_aig, model.deepgate_mig, model.deepgate_xmg, model.deepgate_xag]:
        for param in encoder.parameters():
            param.requires_grad = False
    
    # 冻结transformer参数
    for param in model.mask_tf.parameters():
        param.requires_grad = False
    
    trainer = mixgate.TopTrainer(args, model, distributed=True, device='cuda:0')
    trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0, 2.0], lr=1e-4, lr_step=50)  # prob_loss + alignment_loss
    trainer.train(40, train_dataset, val_dataset)
    
    stage2_ckpt_path = os.path.join(args.save_dir, 'stage2_alignment.pth')
    trainer.save(stage2_ckpt_path)
    
    # 复制日志文件到统一目录
    if hasattr(trainer, 'log_path') and os.path.exists(trainer.log_path):
        log_filename = os.path.basename(trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'stage2_{log_filename}')
        shutil.copy2(trainer.log_path, unified_log_path)
        print(f'[INFO] Stage2 training log saved to: {unified_log_path}')
    return stage2_ckpt_path

def train_stage3_full_model(args, train_dataset, val_dataset, stage2_ckpt_path):
    """
    第三阶段：训练完整的模型（包括transformer和mask）
    目标：所有损失（prob_loss + mcm_loss + func_loss + alignment_loss）
    """
    print('[INFO] ===== Stage 3: Training Full Model =====')
    
    # 加载第二阶段的模型
    model = mixgate.TopModel(args, '', '', '', '')  # 临时创建，会被load覆盖
    trainer = mixgate.TopTrainer(args, model, distributed=True, device='cuda:0')
    trainer.load(stage2_ckpt_path)
    
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    
    # 设置完整的损失权重
    trainer.set_training_args(loss_weight=[3.0, 1.0, 0.5, 1.0], lr=5e-5, lr_step=50)  # 降低学习率
    trainer.train(60, train_dataset, val_dataset)
    
    final_ckpt_path = os.path.join(args.save_dir, 'final_model.pth')
    trainer.save(final_ckpt_path)
    
    # 复制日志文件到统一目录
    if hasattr(trainer, 'log_path') and os.path.exists(trainer.log_path):
        log_filename = os.path.basename(trainer.log_path)
        unified_log_path = os.path.join(args.save_dir, f'stage3_{log_filename}')
        shutil.copy2(trainer.log_path, unified_log_path)
        print(f'[INFO] Stage3 training log saved to: {unified_log_path}')
    return final_ckpt_path

if __name__ == '__main__':
    args = get_parse_args()
    
    print('[INFO] Parse Dataset')
    train_dataset, val_dataset = load_preprocessed_data(DATA_DIR, 'merge_consistency15001%_inmemory')
    
    # 第一阶段：训练4个GNN编码器
    encoder_ckpts = train_stage1_encoders(args, train_dataset, val_dataset)
    
    # 第二阶段：训练编码器对齐
    stage2_ckpt = train_stage2_alignment(args, train_dataset, val_dataset, encoder_ckpts)
    
    # 第三阶段：训练完整模型
    final_ckpt = train_stage3_full_model(args, train_dataset, val_dataset, stage2_ckpt)
    
    print('[INFO] ===== Training Complete =====')
    print(f'[INFO] Final model saved to: {final_ckpt}')
