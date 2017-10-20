
DATASET_DIR=/home/weiliu/data/datasets/mio-tcd/tfrecord/miotcd_classification_tfrecord
TRAIN_DIR=/media/weiliu/data/results/miotcd/classification/models/resnet_v2_101_ls
DATASET_DIR=/media/weiliu/data/datasets/mio-tcd/tfrecord/miotcd_classification_tfrecord_test
EVAL_DIR= /home/weiliu/data/results/miotcd/classification/evaluation/resnet_v2_101_ls
CHECKPOINT_PATH=/home/weiliu/data/models/tensorflow/slim/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt
NUM_CLASSES=11
NUM_SAMPLES=519164
WEIGHT_FLAG=True
MODEL_NAME=resnet_v2_101
LABELS_TO_NAMES_PATH=../datasets/miotcd_labels.txt
python ../train_image_classifier_miotcd.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --num_classes=${NUM_CLASSES} \
    --labels_to_names_path=${LABELS_TO_NAMES_PATH} \
    --num_samples=${NUM_SAMPLES} \
    --learning_rate=0.001 \
    --label_smoothing=0.1 \
    --end_learning_rate=0.00001 \
    --dataset_split_name=train \
    --batch_size=128 \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --max_number_of_steps=80000 \
    --checkpoint_exclude_scopes=${MODEL_NAME}/logits
##    --weights_flag=${WEIGHTS_FLAG} \


python ../eval_image_classifier_miotcd.py \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR}\
    --model_name=${MODEL_NAME}
