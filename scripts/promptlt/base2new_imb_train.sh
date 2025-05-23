#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/your/data"
TRAINER=PromptLT


DATASET=$1
EPOCHS=$2
IMB=$3

CFG=imb_vit_b16
SHOTS=$4

for SEED in 1 2 3
do
DIR=/path/to/your/Candle/output/base2new/train_base/${DATASET}/imbratio${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    rm -rf /path/to/your/Candle/output/base2new/train_base/${DATASET}/imbratio${IMB}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
fi
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

done