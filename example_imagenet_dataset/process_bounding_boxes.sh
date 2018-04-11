#Convert the XML files for bounding box annotations into a single CSV.
WORK_DIR=/workspace/zhangbin/master/models/research/inception/inception
SCRATCH_DIR=/workspace/zhangbin/dataset_robin/imagenet-data/raw-data
LABELS_FILE="${WORK_DIR}/data/imagenet_lsvrc_2015_synsets.txt"
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="${WORK_DIR}/data/process_bounding_boxes.py"
BOUNDING_BOX_FILE="${SCRATCH_DIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${SCRATCH_DIR}/ILSVRC2012_bbox_train_v2/"

"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort > "${BOUNDING_BOX_FILE}"
echo "Finished downloading and preprocessing the ImageNet data."
