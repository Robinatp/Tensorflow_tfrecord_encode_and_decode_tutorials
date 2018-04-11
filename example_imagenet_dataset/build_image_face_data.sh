#Note the locations of the train and validation data.
TRAIN_DIR=/workspace/zhangbin/dataset_robin/face/train/

VALIDATION_DIR=/workspace/zhangbin/dataset_robin/face/validation/
LABELS_FILE=/workspace/zhangbin/dataset_robin/face/labels.txt

# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=/workspace/zhangbin/dataset_robin/face/tfrecord/

python inception/data/build_image_data.py \
  --train_directory=${TRAIN_DIR} \
  --validation_directory=${VALIDATION_DIR} \
  --output_directory=${OUTPUT_DIRECTORY} \
  --labels_file=${LABELS_FILE} \
  --train_shards=128 \
  --validation_shards=32 \
  --num_threads=8
