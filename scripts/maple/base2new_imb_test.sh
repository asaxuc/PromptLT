#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/your/data"
TRAINER=MaPLe

DATASET=$1
EPOCHS=$2
IMB=$3

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=$4
LOADEP=50
SUB=$5

for SEED in 1 # 2 3
do
COMMON_DIR=${DATASET}/imbratio1${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/path/to/your/Candle/output/base2new/train_base/${COMMON_DIR}
DIR=/path/to/your/Candle/output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
  echo "Oops! The results exist at ${DIR} (so skip this job)"
fi 
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    DATASET.IMBALANCE_RATIO ${IMB} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.PHASE "test" 
done