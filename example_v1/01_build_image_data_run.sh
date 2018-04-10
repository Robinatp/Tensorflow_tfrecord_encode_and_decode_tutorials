#!/bin/bash
## Note the locations of the train and validation data.
TRAIN_DIR=`pwd`/gestures-dataset/train/

VALIDATION_DIR=`pwd`/gestures-dataset/validation/
LABELS_FILE=`pwd`/gestures-dataset/labels.txt

# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=`pwd`/gestures-dataset/

python 01_build_image_data.py \
  --train_directory=${TRAIN_DIR} \
  --validation_directory=${VALIDATION_DIR} \
  --output_directory=${OUTPUT_DIRECTORY} \
  --labels_file=${LABELS_FILE} \
  --train_shards=4 \
  --validation_shards=4 \
  --num_threads=2
