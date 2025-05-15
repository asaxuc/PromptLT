#!/bin/bash

#cd ../..

# custom config
DATA="/data/wxc/TIP-data"
TRAINER=MMRL


DATASET=$1
EPOCHS=$2
IMB=$3

CFG=vit_b16
SHOTS=$4

for SEED in 1 # 2 3
do
DIR=/data/wxc/remote/Candle/output/base2new/train_base/${DATASET}/imbratio111${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    rm -rf /data/wxc/remote/Candle/output/base2new/train_base/${DATASET}/imbratio111${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
fi
#    echo "Results are available in ${DIR}. Resuming..."
#    python train.py \
#    --root ${DATA} \
#    --seed ${SEED} \
#    --trainer ${TRAINER} \
#    --dataset-config-file configs/datasets/${DATASET}.yaml \
#    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#    --output-dir ${DIR} \
#    OPTIM.MAX_EPOCH ${EPOCHS} \
#    DATASET.IMBALANCE_RATIO ${IMB} \
#    DATASET.NUM_SHOTS ${SHOTS} \
#    DATASET.SUBSAMPLE_CLASSES base \
#    TRAINER.PHASE "train"
#else
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
    TASK B2N \
    TRAINER.PHASE "train"
#fi
done