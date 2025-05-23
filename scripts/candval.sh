for SEED in 1 2 3
do
COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/path/to/your/Candle/output/base2new/train_base/${COMMON_DIR}
DIR=/path/to/your/Candle/output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
  echo "Oops! The results exist at ${DIR} (so skip this job)"
   echo "Evaluating model"
   echo "Results are available in ${DIR}. Resuming..."

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
fi
