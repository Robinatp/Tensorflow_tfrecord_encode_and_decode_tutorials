DATASET_DIR=/home/zhangbin/GitHub/SSD-Tensorflow/VOC/VOC2007/VOCdevkit_train/VOC2007/
OUTPUT_DIR=`pwd`/tfrecords
echo ${OUTPUT_DIR}
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
