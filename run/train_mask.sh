NUM_PROC=2
GPUS=0,1
MASK=0.03


python -m torchrun --nproc_per_node=1 --master_port=29964 train_mask.py \
 --exp_id mcm_0.03 \
 --batch_size 4 --num_epochs 60 \
 --mask_ratio 0.03 \
 --gpus 7


# python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29958 train_mask.py \
#  --exp_id mcm_$MASK \
#  --batch_size 8 --num_epochs 60 \
#  --mask_ratio $MASK \
#  --gpus $GPUS
