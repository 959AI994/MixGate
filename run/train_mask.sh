NUM_PROC=4
GPUS=0,1,2,3
MASK=0


python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_mask.py \
 --exp_id mcm_${MASK} \
 --batch_size 4 --num_epochs 60 \
 --hier_tf \
 --mask_ratio ${MASK} \
 --gpus ${GPUS}


# python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=29958 train_mask.py \
#  --exp_id mcm_$MASK \
#  --batch_size 8 --num_epochs 60 \
#  --mask_ratio $MASK \
#  --gpus $GPUS
