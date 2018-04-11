# Note the locations of the train and validation data.
WORK_DIR=/workspace/zhangbin/master/models/research/inception/inception

VALIDATION_DIRECTORY=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data/ILSVRC2012_img_val/

# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.ls

echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="${WORK_DIR}/data/preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${WORK_DIR}/data/imagenet_2012_validation_synset_labels.txt"

"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"
