# MixGate
## debug
torchrun --nproc_per_node=1 --master_port=29958 train_mask.py     --exp_id nomcm_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0    --hier_tf
## train
torchrun --nproc_per_node=4 --master_port=29958 train_mask.py     --exp_id nomcm_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0,4,5,6    --hier_tf

## 若替换gnn为gcn+ae
使用_ae后缀的几个py文件
