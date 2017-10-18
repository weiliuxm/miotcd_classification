DATASET_DIR=/media/Store/weiliu/datasets/MIO-TCD/miotcd_classification_tfrecord
TRAIN_DIR=/media/Store/weiliu/tmp/miotcd-models/resnet_v2_50_w
CHECKPOINT_PATH=/media/Store/weiliu/models/TF/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt
NUM_CLASSES=11
NUM_SAMPLES=519164
MODEL_NAME=resnet_v2_50
LABELS_TO_NAMES_PATH=/media/Store/weiliu/datasets/MIO-TCD/miotcd_classification_tfrecord/labels.txt
WEIGHTS_FLAG=True
python train_image_classifier_miotcd_weighted.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --num_classes=${NUM_CLASSES} \
    --labels_to_names_path=${LABELS_TO_NAMES_PATH} \
    --num_samples=${NUM_SAMPLES} \
    --learning_rate=0.0001 \
    --label_smoothing=0.1 \
    --end_learning_rate=0.00001 \
    --dataset_split_name=train \
    --batch_size=128 \
    --weights_flag=${WEIGHTS_FLAG} \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --max_number_of_steps=80100 \
    --checkpoint_exclude_scopes=resnet_v2_50/logits
#    --trainable_scopes=resnet_v2_50/logits


DATASET_DIR=/media/Store/weiliu/datasets/MIO-TCD/miotcd_classification_tfrecord_test
EVAL_DIR= /media/Store/weiliu/tmp/results/evaluation_w
python eval_image_classifier_miotcd.py \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR}\
    --model_name=${MODEL_NAME}
