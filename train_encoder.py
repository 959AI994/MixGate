from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 1222111
import mixgate
from mixgate.aig_encoder import hoga
from mixgate.aig_encoder import polargate
from mixgate.aig_encoder import deepgate3
from mixgate.aig_encoder import gcn
import torch
import os
from config import get_parse_args
from mixgate.dc_trainer import Trainer

DATA_DIR = '/home/xqgrp/wangjingxin/datasets/mixgate_data'

if __name__ == '__main__':
    args = get_parse_args()
    circuit_path = os.path.join(DATA_DIR, 'merged_all1500.npz')
    num_epochs = 60
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')

    if args.aig_encoder == 'pg':
        model = polargate.PolarGate(args, in_dim=3, out_dim=args.dim_hidden,layer_num=9)
    elif args.aig_encoder == 'dg3':
        model = deepgate3.DeepGate3(dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'gcn':
        model = gcn.DirectMultiGCNEncoder(dim_feature=3, dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'hoga':
        model = hoga.HOGA(in_channels=3, hidden_channels=args.dim_hidden, out_channels=args.dim_hidden, num_layers=1,
                            dropout=0.1, num_hops=5+1, heads=8, directed = True, attn_type="mix")
    
    trainer = Trainer(args, model, distributed=True)
    trainer.set_training_args(loss_weight=[1.0,0.0,0.0], lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    # Save checkpoint after stage 1
    trainer.save(os.path.join(trainer.log_dir, 'stage1_model.pth'))
    # Load stage-1 checkpoint
    print('[INFO] Loading Stage 1 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage1_model.pth'))

    print('[INFO] Stage 2 Training ...')
    trainer.set_training_args(loss_weight = [1.0, 0.0, 1.0], lr=1e-4, lr_step=50)
    trainer.train(num_epochs, train_dataset, val_dataset)
    trainer.save(os.path.join(trainer.log_dir, 'stage2_model.pth'))