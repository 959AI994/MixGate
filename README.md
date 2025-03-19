# MixGate
## debug
torchrun --nproc_per_node=1 --master_port=29958 train_mask.py     --exp_id nomcm_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0    --hier_tf
## train
torchrun --nproc_per_node=4 --master_port=29958 train_mask.py     --exp_id nomcm_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0,4,5,6    --hier_tf

# 训练时添加kl_weight参数
python train.py --kl_weight 0.1 --dim_hidden 128 --batch_size 32
