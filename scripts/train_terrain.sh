MODEL='bicycle_gan'
# dataset details
CLASS='terrain'  # facades, day2night, edges2shoes, edges2handbags, maps
NZ=8
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
FINE_SIZE=256
INPUT_NC=3
NITER=25
NITER_DECAY=25
UPSAMPLE='basic' #'nearest'  or 'basic'  or 'bilinear'
WHERE_ADD='all'

# training
GPU_ID=2
EPOCHS=$((NITER + NITER_DECAY))
DISPLAY_ID=$((GPU_ID*10+1))
CHECKPOINTS_DIR=./checkpoints_pub/${CLASS}_${EPOCHS}_${UPSAMPLE}_${WHERE_ADD}_${NZ}/
NAME=${CLASS}_${MODEL}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  --which_direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout \
  --upsample ${UPSAMPLE} \
  --where_add ${WHERE_ADD} \
  ${NO_FLIP}
