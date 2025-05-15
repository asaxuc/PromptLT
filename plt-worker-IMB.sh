# base-to-new generalization

DATASET=$1
IMB=$3 # imbalance ratio, set to 1.0 for balanced scenarios; fixed at 0.01 for imagenet
EPOCHS=50
SHOTS=$2 # maximum number of samples per class

# training
bash scripts/mmrl/base2new_imb_train.sh ${DATASET}  ${EPOCHS} ${IMB} ${SHOTS} 

bash scripts/mmrl/base2new_imb_test.sh ${DATASET}  ${EPOCHS} ${IMB} ${SHOTS}  base

bash scripts/mmrl/base2new_imb_test.sh ${DATASET}  ${EPOCHS} ${IMB}  ${SHOTS} new
