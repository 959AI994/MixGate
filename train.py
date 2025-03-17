from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 1222111
import mixgate
import torch
import os
from config import get_parse_args

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data'

if __name__ == '__main__':
    args = get_parse_args()
    circuit_path = os.path.join(DATA_DIR, 'merged_all.npz')
    num_epochs = 60
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = mixgate.Model()
    
    trainer = mixgate.Trainer(args, model, distributed=False)
    trainer.set_training_args(loss_weight=[3.0, 1.0, 0.5], lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    