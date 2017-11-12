TRAIN_DIR=/media/weiliu/data/results/miotcd/classification/models/resnet_v2_50_ls_weighted
LABES_TO_NAMES_PATH=/home/weiliu/tools/slim/models/research/slim/datasets/labels.txt
TEST_DIR=/media/weiliu/data/datasets/mio-tcd/images/test
RESULTS_DIR=/media/weiliu/data/results/miotcd/classification/csvResults
MODEL_NAME=resnet_v2_50
NUM_CLASSES=11
python test_image_classifier.py \
    --checkpoint_path=${TRAIN_DIR} \
    --test_dir=${TEST_DIR} \
    --num_classes=11 \
    --results_dir=${RESULTS_DIR} \
    --model_name=${MODEL_NAME}
