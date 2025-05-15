#!/bin/bash

#cd ../..

# custom config
DATA="/data/wxc/TIP-data"
TRAINER=TCP


DATASET=$1
EPOCHS=$2
IMB=$3

CFG=vit_b16_ep100_ctxv1
SHOTS=100

for SEED in 1 # 2 3
do
DIR=/data/wxc/remote/Candle/output/base2new/train_base/${DATASET}/imbratio${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    DATASET.IMBALANCE_RATIO ${IMB} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.PHASE "train"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    DATASET.IMBALANCE_RATIO ${IMB} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.PHASE "train"
fi
done