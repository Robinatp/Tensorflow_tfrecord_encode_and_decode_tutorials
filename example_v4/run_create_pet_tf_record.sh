python create_pet_tf_record.py \
        --data_dir=/home/zhangbin/eclipse-workspace-python/TF_models/src/models/research/object_detection/01_pets_tf_tutorials/dataset \
        --output_dir=/home/zhangbin/GitHub/TFRecord-file/example_v4/output \
        --label_map_path=/home/zhangbin/eclipse-workspace-python/TF_models/src/models/research/object_detection/01_pets_tf_tutorials/dataset/pet_label_map.pbtxt \
        --faces_only=False

python create_pet_tf_record.py \
        --data_dir=/home/zhangbin/eclipse-workspace-python/TF_models/src/models/research/object_detection/01_pets_tf_tutorials/dataset \
        --output_dir=/home/zhangbin/GitHub/TFRecord-file/example_v4/output \
        --label_map_path=/home/zhangbin/eclipse-workspace-python/TF_models/src/models/research/object_detection/01_pets_tf_tutorials/dataset/pet_label_map.pbtxt \
        --faces_only=True