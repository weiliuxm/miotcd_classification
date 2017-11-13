TRAIN_DIR=/media/weiliu/data/results/miotcd/classification/models/resnet_v2_50_ls_weighted
#LABES_TO_NAMES_PATH=/home/weiliu/tools/slim/models/research/slim/datasets/labels.txt
TEST_DIR=/media/weiliu/data/datasets/mio-tcd/images/test
RESULTS_DIR=/media/weiliu/data/results/miotcd/classification/csvResults
MODEL_NAME=resnet_v2_50
PREFIX_CSV=resnet_v2_50_ls_weighted
NUM_CLASSES=11
python test_image_classifier_miotcd.py \
    --checkpoint_path=${TRAIN_DIR} \
    --test_dir=${TEST_DIR} \
    --prefix_csv=${PREFIX_CSV} \
    --num_classes=11 \
    --results_dir=${RESULTS_DIR} \
    --model_name=${MODEL_NAME}

