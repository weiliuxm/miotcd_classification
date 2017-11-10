DATASET_DIR=/media/weiliu/data/datasets/mio-tcd/tfrecord/miotcd_classification_tfrecord
TRAIN_DIR=/media/weiliu/data/results/miotcd/classification/models/inception_v3
CHECKPOINT_PATH=/media/weiliu/data/models/tensorflow/slim/inception_v3.ckpt
NUM_CLASSES=11
NUM_SAMPLES=519164
WEIGHT_FLAG=False
MODEL_NAME=inception_v3
LABELS_TO_NAMES_PATH=/media/weiliu/data/datasets/mio-tcd/tfrecord/labels.txt
python train_image_classifier_miotcd.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --num_classes=${NUM_CLASSES} \
    --labels_to_names_path=${LABELS_TO_NAMES_PATH} \
    --num_samples=${NUM_SAMPLES} \
    --learning_rate=0.001 \
    --end_learning_rate=0.00001 \
    --label_smoothing=0 \
    --weights_flag=${WEIGHTS_FLAG} \
    --dataset_split_name=train \
    --batch_size=64 \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --max_number_of_steps=80000 \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits

#DATASET_DIR=/media/weiliu/data/datasets/mio-tcd/tfrecord/miotcd_classification_tfrecord_test
#EVAL_DIR= /media/weiliu/data/results/miotcd/classification/eva/inception_v3
#python eval_image_classifier_miotcd.py \
#    --checkpoint_path=${TRAIN_DIR} \
#    --eval_dir=${EVAL_DIR} \
#    --dataset_split_name=validation \
#    --dataset_dir=${DATASET_DIR}\
#    --model_name=${MODEL_NAME}
