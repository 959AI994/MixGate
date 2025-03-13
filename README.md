# MixGate

torchrun --nproc_per_node=6 --master_port=29958 train_mask.py     --exp_id mcm_0.03     --batch_size 4     --num_epochs 60     --mask_ratio 0.03     --gpus 0,1,4,5,6,7

torchrun --nproc_per_node=1 --master_port=29958 train_mask.py     --exp_id mcm_0.03     --batch_size 4     --num_epochs 60     --mask_ratio 0.03     --gpus 0

python -m torch.distributed.launch --nproc_per_node=5 --master_port=29964 train_mask.py --exp_id mcm_0.03 --batch_size 4 --num_epochs 60 --mask_ratio 0.03 --gpus 0,1,2,3,7