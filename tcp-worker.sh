# base-to-new generalization

DATASET=$1
IMB=$2 # imbalance ratio, set to 1.0 for balanced scenarios; fixed at 0.01 for imagenet
EPOCHS=50
SHOTS=100 # maximum number of samples per class

# training
bash scripts/tcp/base2new_imb_train.sh ${DATASET}  ${EPOCHS} ${IMB}

bash scripts/tcp/base2new_imb_test.sh ${DATASET}  ${EPOCHS} ${IMB} base

bash scripts/tcp/base2new_imb_test.sh ${DATASET}  ${EPOCHS} ${IMB} new
