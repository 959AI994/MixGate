# MixGate
This repo is another version of MixGate, which masks AIG and thereby enhances the representation of AIG with the help of MixGate.

# Parser
torchrun --nproc_per_node=1 --master_port=29918 train_mask.py     --exp_id 01_polargate_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 7    --hier_tf  --aig_encoder pg

# Terminal

## Pretrain MixGate-DeepGate2
torchrun --nproc_per_node=8 --master_port=27710 train_mask.py     --exp_id 01_deepgate2_0.03_alignment     --batch_size 16     --num_epochs 120     --mask_ratio 0.03     --gpus 0,1,2,3,4,5,6,7    --hier_tf  --aig_encoder dg2

nohup torchrun --nproc_per_node=5 --master_port=27710 train_mask.py     --exp_id 01_deepgate2_0.01_alignment_final_2     --batch_size 8     --num_epochs 180     --mask_ratio 0.01     --gpus 0,1,2,3,4    --hier_tf  --aig_encoder dg2

nohup torchrun --nproc_per_node=2 --master_port=28810 train_mask.py \
    --exp_id 01_deepgate2_0.01_alignment_15000_1% \
    --batch_size 4 \
    --mask_ratio 0.01 \
    --gpus 0,1 \
    --hier_tf \
    --aig_encoder dg2

torchrun --nproc_per_node=2 --master_port=27710 train_mask.py \
    --exp_id 01_deepgate2_0.01_no_alignment \
    --batch_size 8 \
    --mask_ratio 0.01 \
    --gpus 0,1 \
    --hier_tf \
    --aig_encoder dg2 \
    --disable_alignment
    

## Pretrain MixGate-PolarGate
torchrun --nproc_per_node=3 --master_port=29918 train_mask.py     --exp_id 01_polargate_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 2,3,4    --hier_tf  --aig_encoder pg

### Resume MixGate-PolarGate
torchrun --nproc_per_node=3 --master_port=29918 train_polargate_checkpoint.py     --exp_id 01_polargate_0.00     --batch_size 4     --num_epochs 500     --mask_ratio 0.00     --gpus 2,3,4    --hier_tf  --aig_encoder pg

## Pretrain MixGate-HOGA
torchrun --nproc_per_node=3 --master_port=28818 train_mask.py     --exp_id 01_hoga_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 5,6,7    --hier_tf  --aig_encoder hoga

### Resume MixGate-HOGA
torchrun --nproc_per_node=3 --master_port=28818 train_hoga_checkpoint.py     --exp_id 01_hoga_0.00     --batch_size 4     --num_epochs 320     --mask_ratio 0.00     --gpus 5,6,7    --hier_tf  --aig_encoder hoga 

## Pretrain MixGate-GCN
torchrun --nproc_per_node=2 --master_port=28818 train_mask.py     --exp_id 01_gcn_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 4,7    --hier_tf  --aig_encoder gcn

## Pretrain MixGate-dg3
torchrun --nproc_per_node=2 --master_port=27718 train_mask.py     --exp_id 01_dg3_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0,7    --hier_tf  --aig_encoder dg3


# Original Encoder
If you want to train an original encoder，you can run this script.
## HOGA Encoder
torchrun --nproc_per_node=1 --master_port=29918 train_encoder.py --gpus 0 --aig_encoder hoga
## PolarGate Encoder
torchrun --nproc_per_node=2 --master_port=29918 train_encoder.py --gpus 4,7 --aig_encoder pg --batch_size 256
## GCN Encoder
torchrun --nproc_per_node=1 --master_port=26618 train_encoder.py --gpus 0 --aig_encoder gcn --batch_size 4 --exp_id 02_origin_gcn
## DeepGate3 Encoder
torchrun --nproc_per_node=4 --master_port=25518 train_encoder.py --gpus 2,3,5,6 --aig_encoder dg3 --batch_size 4 --exp_id 02_origin_dg3


# Mem and Runtime

