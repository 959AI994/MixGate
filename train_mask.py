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


# # 获取 local_rank 环境变量，代表当前进程的 GPU
# local_rank = int(os.environ['LOCAL_RANK'])

# # 设置每个进程使用的 GPU
# torch.cuda.set_device(local_rank)
# device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# # 初始化分布式训练
# dist.init_process_group(backend='nccl', init_method='env://')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/dg_pair'

if __name__ == '__main__':
    args = get_parse_args()
    # here,we need to build some npz formate including mig,xmg,xag,aig fusion graph

    # circuit_path = 'datasets/pair_graphs.npz'
    circuit_path ='/home/jwt/1/150/merged_all150.npz'
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)

    train_dataset, val_dataset = dataset.get_dataset()
    # for data in train_dataset:
    #     # print(f"edge_index: {data.edge_index}")
    #     # print(f"num_nodes: {data.num_nodes}")
    #     # print(f"name: {data.name}")
    #     assert data.edge_index.min() >= 0, "edge_index 包含负数索引"
    #     assert data.edge_index.max() < data.num_nodes, "edge_index 包含超出范围的索引"
    print('[INFO] Create Model and Trainer')
    model = mixgate.top_model.TopModel(
        args, 
        # dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt_aig='/home/jwt/MixGate/ckpt/model_func_aig.pth',
        dg_ckpt_xag='/home/jwt/MixGate/ckpt/model_func_xag.pth',
        dg_ckpt_xmg='/home/jwt/MixGate/ckpt/model_func_xmg.pth',
        dg_ckpt_mig='/home/jwt/MixGate/ckpt/model_func_mig.pth'
    )
    
    # model.to(device)  # 将模型移到正确的设备

    # # 使用 DistributedDataParallel
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed=True)
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight = [3.0, 1.0, 0.0])
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight = [3.0, 1.0, 3.0])
    print('[INFO] Stage 2 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)


    # print('[INFO] Stage 3 Training ...')
    # trainer.set_training_args(loss_weight = [0.0, 0.0, 1.0], lr=1e-4, lr_step=50)
    # trainer.train(num_epochs, train_dataset, val_dataset)

    
    