from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
import argparse
from config import get_parse_args

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data'

class StagedTrainingConfig:
    """分阶段训练配置类"""
    
    def __init__(self):
        # 第一阶段：GNN编码器训练配置
        self.stage1_config = {
            'epochs': 60,
            'lr': 1e-4,
            'lr_step': 50,
            'batch_size': 16,  # 可以根据GPU内存调整
            'loss_weights': {
                'prob_loss': 1.0,
                'mcm_loss': 0.0,
                'tt_loss': 1.0,
                'alignment_loss': 0.0
            }
        }
        
        # 第二阶段：编码器对齐训练配置
        self.stage2_config = {
            'epochs': 60,
            'lr': 1e-4,
            'lr_step': 50,
            'batch_size': 16,  
            'loss_weights': {
                'prob_loss': 1.0,
                'mcm_loss': 0.0,
                'tt_loss': 0.0,
                'alignment_loss': 2.0
            },
            'freeze_encoders': True,
            'freeze_transformer': True
        }
        
        # 第三阶段：完整模型训练配置
        self.stage3_config = {
            'epochs': 60,
            'lr': 5e-5,
            'lr_step': 50,
            'batch_size': 8,  
            'loss_weights': {
                'prob_loss': 3.0,
                'mcm_loss': 1.0,
                'tt_loss': 0.5,
                'alignment_loss': 1.0
            },
            'freeze_encoders': False,
            'freeze_transformer': False
        }

def create_encoder_model(args, encoder_type):
    """根据类型创建编码器模型"""
    if encoder_type == 'aig':
        if args.aig_encoder == 'dg2':
            return mixgate.dg_model.Model(dim_hidden=args.dim_hidden)
        elif args.aig_encoder == 'pg':
            return mixgate.aig_encoder.polargate.PolarGate(args, in_dim=3, out_dim=args.dim_hidden)
        elif args.aig_encoder == 'dg3':
            return mixgate.aig_encoder.deepgate3.DeepGate3(dim_hidden=args.dim_hidden)
        elif args.aig_encoder == 'gcn':
            return mixgate.aig_encoder.gcn.DirectMultiGCNEncoder(dim_feature=3, dim_hidden=args.dim_hidden)
        elif args.aig_encoder == 'hoga':
            return mixgate.aig_encoder.hoga.HOGA(in_channels=3, hidden_channels=args.dim_hidden, 
                                                out_channels=args.dim_hidden, num_layers=1,
                                                dropout=0.1, num_hops=5+1, heads=8, directed=True, attn_type="mix")
    elif encoder_type == 'mig':
        return mixgate.dg_model_mig.Model(dim_hidden=args.dim_hidden)
    elif encoder_type == 'xmg':
        return mixgate.dg_model_xmg.Model(dim_hidden=args.dim_hidden)
    elif encoder_type == 'xag':
        return mixgate.dg_model_xag.Model(dim_hidden=args.dim_hidden)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def train_single_encoder(args, encoder_type, train_dataset, val_dataset, config):
    """训练单个编码器"""
    print(f'[INFO] Training {encoder_type.upper()} Encoder...')
    
    # 临时修改args的batch_size
    original_batch_size = args.batch_size
    args.batch_size = config['batch_size']
    print(f'[INFO] Using batch_size: {args.batch_size}')
    
    model = create_encoder_model(args, encoder_type)
    trainer = mixgate.dc_trainer.Trainer(args, model, distributed=False, device='cuda:0')
    
    # 设置训练参数
    loss_weights = [config['loss_weights']['prob_loss'], 
                   config['loss_weights']['mcm_loss'], 
                   config['loss_weights']['tt_loss']]
    
    trainer.set_training_args(loss_weight=loss_weights, 
                             lr=config['lr'], 
                             lr_step=config['lr_step'])
    
    trainer.train(config['epochs'], train_dataset, val_dataset)
    
    # 恢复原始batch_size
    args.batch_size = original_batch_size
    
    # 保存检查点
    ckpt_path = os.path.join(trainer.log_dir, f'{encoder_type}_encoder_stage1.pth')
    trainer.save(ckpt_path)
    
    return ckpt_path

def train_stage1_encoders(args, train_dataset, val_dataset, config):
    """第一阶段：训练4个GNN编码器"""
    print('[INFO] ===== Stage 1: Training 4 GNN Encoders =====')
    print('[INFO] Target: Alignment Loss + Probability Loss + TT Loss')
    
    encoder_types = ['aig', 'mig', 'xmg', 'xag']
    encoder_ckpts = {}
    
    for encoder_type in encoder_types:
        ckpt_path = train_single_encoder(args, encoder_type, train_dataset, val_dataset, config)
        encoder_ckpts[encoder_type] = ckpt_path
        print(f'[INFO] {encoder_type.upper()} encoder saved to: {ckpt_path}')
    
    return encoder_ckpts

def train_stage2_alignment(args, train_dataset, val_dataset, encoder_ckpts, config):
    """第二阶段：训练编码器对齐"""
    print('[INFO] ===== Stage 2: Training Encoder Alignment =====')
    print('[INFO] Target: Alignment Loss + Probability Loss')
    
    # 临时修改args的batch_size
    original_batch_size = args.batch_size
    args.batch_size = config['batch_size']
    print(f'[INFO] Using batch_size: {args.batch_size}')
    
    # 创建TopModel并加载预训练的编码器
    model = mixgate.TopModel(args, 
                            encoder_ckpts['aig'], 
                            encoder_ckpts['mig'], 
                            encoder_ckpts['xmg'], 
                            encoder_ckpts['xag'])
    
    # 根据配置冻结参数
    if config['freeze_encoders']:
        for encoder in [model.deepgate_aig, model.deepgate_mig, model.deepgate_xmg, model.deepgate_xag]:
            for param in encoder.parameters():
                param.requires_grad = False
        print('[INFO] Encoders frozen')
    
    if config['freeze_transformer']:
        for param in model.mask_tf.parameters():
            param.requires_grad = False
        print('[INFO] Transformer frozen')
    
    trainer = mixgate.TopTrainer(args, model, distributed=False, device='cuda:0')
    
    # 恢复原始batch_size
    args.batch_size = original_batch_size
    
    # 设置训练参数
    loss_weights = [config['loss_weights']['prob_loss'], 
                   config['loss_weights']['mcm_loss'], 
                   config['loss_weights']['tt_loss'], 
                   config['loss_weights']['alignment_loss']]
    
    trainer.set_training_args(loss_weight=loss_weights, 
                             lr=config['lr'], 
                             lr_step=config['lr_step'])
    
    trainer.train(config['epochs'], train_dataset, val_dataset)
    
    stage2_ckpt_path = os.path.join(trainer.log_dir, 'stage2_alignment.pth')
    trainer.save(stage2_ckpt_path)
    
    return stage2_ckpt_path

def train_stage3_full_model(args, train_dataset, val_dataset, stage2_ckpt_path, config):
    """第三阶段：训练完整的模型"""
    print('[INFO] ===== Stage 3: Training Full Model =====')
    print('[INFO] Target: All Losses (Prob + MCM + TT + Alignment)')
    
    # 临时修改args的batch_size
    original_batch_size = args.batch_size
    args.batch_size = config['batch_size']
    print(f'[INFO] Using batch_size: {args.batch_size}')
    
    # 加载第二阶段的模型
    model = mixgate.TopModel(args, '', '', '', '')  # 临时创建
    trainer = mixgate.TopTrainer(args, model, distributed=False, device='cuda:0')
    trainer.load(stage2_ckpt_path)
    
    # 恢复原始batch_size
    args.batch_size = original_batch_size
    
    # 根据配置解冻参数
    if not config['freeze_encoders']:
        for encoder in [model.deepgate_aig, model.deepgate_mig, model.deepgate_xmg, model.deepgate_xag]:
            for param in encoder.parameters():
                param.requires_grad = True
        print('[INFO] Encoders unfrozen')
    
    if not config['freeze_transformer']:
        for param in model.mask_tf.parameters():
            param.requires_grad = True
        print('[INFO] Transformer unfrozen')
    
    # 设置训练参数
    loss_weights = [config['loss_weights']['prob_loss'], 
                   config['loss_weights']['mcm_loss'], 
                   config['loss_weights']['tt_loss'], 
                   config['loss_weights']['alignment_loss']]
    
    trainer.set_training_args(loss_weight=loss_weights, 
                             lr=config['lr'], 
                             lr_step=config['lr_step'])
    
    trainer.train(config['epochs'], train_dataset, val_dataset)
    
    final_ckpt_path = os.path.join(trainer.log_dir, 'final_model.pth')
    trainer.save(final_ckpt_path)
    
    return final_ckpt_path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Staged Training Configuration')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['stage1', 'stage2', 'stage3', 'all'],
                       help='Which stage to run')
    parser.add_argument('--load_stage2', type=str, default='',
                       help='Path to stage2 checkpoint for stage3 training')
    parser.add_argument('--load_encoders', type=str, default='',
                       help='Path to encoder checkpoints directory for stage2 training')
    
    args_main = parser.parse_args()
    args = get_parse_args()
    
    print('[INFO] Parse Dataset')
    # 使用现有的数据加载逻辑
    dataset = mixgate.NpzParser_Pair(DATA_DIR, 'dummy.npz')
    train_dataset, val_dataset = dataset.get_dataset()
    
    # 创建训练配置
    config = StagedTrainingConfig()
    
    if args_main.stage == 'stage1' or args_main.stage == 'all':
        # 第一阶段：训练4个GNN编码器
        encoder_ckpts = train_stage1_encoders(args, train_dataset, val_dataset, config.stage1_config)
        
        if args_main.stage == 'stage1':
            print('[INFO] Stage 1 completed. Encoder checkpoints:')
            for encoder_type, ckpt_path in encoder_ckpts.items():
                print(f'  {encoder_type}: {ckpt_path}')
            return
    
    if args_main.stage == 'stage2' or args_main.stage == 'all':
        # 第二阶段：训练编码器对齐
        if args_main.stage == 'stage2' and args_main.load_encoders:
            # 从指定目录加载编码器检查点
            encoder_ckpts = {}
            for encoder_type in ['aig', 'mig', 'xmg', 'xag']:
                ckpt_path = os.path.join(args_main.load_encoders, f'{encoder_type}_encoder_stage1.pth')
                if os.path.exists(ckpt_path):
                    encoder_ckpts[encoder_type] = ckpt_path
                else:
                    raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")
        else:
            # 使用第一阶段训练的编码器
            encoder_ckpts = train_stage1_encoders(args, train_dataset, val_dataset, config.stage1_config)
        
        stage2_ckpt = train_stage2_alignment(args, train_dataset, val_dataset, encoder_ckpts, config.stage2_config)
        
        if args_main.stage == 'stage2':
            print(f'[INFO] Stage 2 completed. Checkpoint saved to: {stage2_ckpt}')
            return
    
    if args_main.stage == 'stage3' or args_main.stage == 'all':
        # 第三阶段：训练完整模型
        if args_main.stage == 'stage3':
            if args_main.load_stage2:
                stage2_ckpt = args_main.load_stage2
            else:
                raise ValueError("For stage3 training, please provide --load_stage2 checkpoint path")
        else:
            # 使用第二阶段训练的检查点
            stage2_ckpt = train_stage2_alignment(args, train_dataset, val_dataset, encoder_ckpts, config.stage2_config)
        
        final_ckpt = train_stage3_full_model(args, train_dataset, val_dataset, stage2_ckpt, config.stage3_config)
        
        print('[INFO] ===== Training Complete =====')
        print(f'[INFO] Final model saved to: {final_ckpt}')

if __name__ == '__main__':
    main()
