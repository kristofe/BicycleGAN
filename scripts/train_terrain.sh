#!/usr/bin/env bash

# Example usage
# bash scripts/train_terrain.sh -g 2 -d terrain_2560/aligned/ -z 2 -i 150

# 8/31/2018  good setting is LAMBDA_L1 = 20 --use_features lambda_features 1e-6 on paper dataset. try 500 epochs. (250 looked alright)

while getopts g:d:z:i:n:u: option; do
    case "${option}" in
        g) GPU="${OPTARG}"
        ;;
        d) DIR="${OPTARG}"
        ;;
        z) Z_DIM="${OPTARG}"
        ;;
        i) NUM_ITER="${OPTARG}"
        ;;
        n) PNAME="${OPTARG}"
        ;;
        u) UPS="${OPTARG}"
        ;;
        f) LF="${OPTARG}"
        ;;
    esac
done

echo "GPU: ${GPU} datadir: ${DIR} z length: ${Z_DIM}  iter count: ${NUM_ITER} name: ${PNAME} upsample: ${UPS} lamb feat: ${LF}"


MODEL='bicycle_gan'
CLASS='terrain'  # facades, day2night, edges2shoes, edges2handbags, maps

# Default values for parameters that aren't passed
[[ ! -z "${GPU}" ]] && GPU_ID=${GPU} || GPU_ID=0
[[ ! -z "${DIR}" ]] && DATADIR=${DIR} || DATADIR='paper'
[[ ! -z "${Z_DIM}" ]] && NZ=${Z_DIM} || NZ=2
[[ ! -z "${NUM_ITER}" ]] && NITER=${NUM_ITER} || NITER=50
[[ ! -z "${NUM_ITER}" ]] && NITER_DECAY=${NUM_ITER} || NITER_DECAY=50
[[ ! -z "${UPS}" ]] && UPSAMPLE=${UPS} || UPSAMPLE='basic'
[[ ! -z "${LF}" ]] && LAMBDAF=${LF} || LAMBDAF=0.000001
[[ ! -z "${PNAME}" ]] && NAME=${PNAME} || NAME=${CLASS}_${MODEL}_${UPS}_${DATADIR}

echo "GPU_ID: ${GPU_ID} datadir: ${DATADIR} z length: ${NZ}  iter count: ${NITER} ${NITER_DECAY} upsample: ${UPSAMPLE} lambda feat: ${LAMBDAF} name: ${NAME}"

#NZ=2
#NO_FLIP='--no_flip'
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=256
FINE_SIZE=256
INPUT_NC=3
#NITER=150
#NITER_DECAY=150
#UPSAMPLE='basic' #'nearest'  or 'basic'  or 'bilinear'
WHERE_ADD='all'
CONDITIONAL_D='--conditional_D'
LAMBDA_L1=20 # default is 10
USE_L2='--use_L2'
USE_NORMALS='--use_normals'
USE_FEATURES='--use_features'
WHICH_MODEL_NETD='basic_256'
WHICH_MODEL_NETD2='basic_256'

# training
#GPU_ID=2
#DATADIR='terrain_2560/aligned'
#DATADIR='terrain'
EPOCHS=$((NITER + NITER_DECAY))
DISPLAY_ID=$((GPU_ID*10+1))
CHECKPOINTS_DIR=checkpoints_pub/${CLASS}_${EPOCHS}_${UPSAMPLE}_${WHERE_ADD}_${NZ}/
#NAME=${CLASS}_${MODEL}_${DATADIR}


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${DATADIR} \
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
  ${NO_FLIP} \
  ${CONDITIONAL_D} \
  --lambda_L1 ${LAMBDA_L1} \
  ${USE_L2} \
  ${USE_NORMALS} \
  ${USE_FEATURES} \
  --lambda_features ${LAMBDAF} \
  --which_model_netD ${WHICH_MODEL_NETD} \
  --which_model_netD2 ${WHICH_MODEL_NETD2}
