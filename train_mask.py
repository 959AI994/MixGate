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

    # Default: merged circuits .npz under ./data (override with env for your setup)
    circuit_path = os.environ.get(
        'MIXGATE_CIRCUIT_NPZ',
        os.path.join(DATA_DIR, 'merged_mixgate1%1500.npz'),
    )
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
    
    # Stage 1: 60 epochs, prob only (no alignment loss)
    print('[INFO] Stage 1 Training: Prob training only (60 epochs) ...')
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1.0, 0.1, 0.0, 0.1])
    trainer.train(60, train_dataset, val_dataset)
    
    # Save checkpoint after stage 1
    trainer.save(os.path.join(trainer.log_dir, 'stage1_prob_only_model.pth'))
    
    # Load stage-1 checkpoint
    print('[INFO] Loading Stage 1 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage1_prob_only_model.pth'))

    # Stage 2: 60 epochs, prob + func (no alignment loss)
    print('[INFO] Stage 2 Training: Prob and Func training (60 epochs) ...')
    trainer.set_training_args(lr=1e-3, lr_step=50, loss_weight=[1.0, 0.1, 1.0, 0.1])
    trainer.train(60, train_dataset, val_dataset)
    
    # Save checkpoint after stage 2
    trainer.save(os.path.join(trainer.log_dir, 'stage2_prob_func_model.pth'))
    
    # Load stage-2 checkpoint
    print('[INFO] Loading Stage 2 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage2_prob_func_model.pth'))

    # Stage 3: final 60 epochs, prob + func + mask (no alignment loss)
    print('[INFO] Stage 3 Training: Prob, Func, and Mask training (60 epochs) ...')
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1.0, 0.0, 1.0, 0.0])
    trainer.train(60, train_dataset, val_dataset)
    
    # Save final model
    trainer.save(os.path.join(trainer.log_dir, 'stage3_final_model.pth'))
    
    print('[INFO] Training completed!')
    print('[INFO] Stage 1: Prob training only (60 epochs)')
    print('[INFO] Stage 2: Prob and Func training (60 epochs)')
    print('[INFO] Stage 3: Prob, Func, and Mask training (60 epochs)')
    