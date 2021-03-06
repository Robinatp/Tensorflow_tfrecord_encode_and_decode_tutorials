#coding=utf-8
import tensorflow as tf

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm

slim = tf.contrib.slim

NUM_CLASSES = 1000
SPLITS_TO_SIZES = {
    'train': 5011,
    'val': 4952,
}
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

labels_to_class =['none','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand',
                  'hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand'
                  'hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand','hand']

FILE_PATTERN = 'coco_%s.record'

def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/%s*.tfrecord' % (dataset_dir, split_name)

def colors_subselect(colors, num_classes=1):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

def bboxes_draw_on_img(img, classes, bboxes, colors, thickness=1):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s' % (classes[i])
        p1 = (p1[0]+15, p1[1]+5)
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.
    Returns:
      A `Dataset` namedtuple.
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)
    
    if not file_pattern:
        file_pattern = FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    print(file_pattern)
    
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    
    # 适配器1：将example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }
    
    #适配器2：将反序列化的数据组装成更高级的格式。由slim完成
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }
    # 解码器
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    
    # dataset对象定义了数据集的文件位置，解码方式等元信息
    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=reader,
                num_samples = SPLITS_TO_SIZES['val'], # 手动生成了三个文件， 每个文件里只包含一个example
                decoder=decoder,
                items_to_descriptions = ITEMS_TO_DESCRIPTIONS,
                num_classes=NUM_CLASSES)
    return dataset


def test():
    dataset = get_split('val', "output")
    #provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers=3,
                        common_queue_capacity=20 * 1,
                        common_queue_min=10 * 1,
                        shuffle=True)
    

    for item in provider._items_to_tensors:
        print(item, provider._items_to_tensors[item])
        
    [image,  glabels, gbboxes] = provider.get(['image', 
                                               'object/label',
                                               'object/bbox'])
    
    colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=100)
    
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('Start verification process...',provider._num_samples)
        for l in range(provider._num_samples):
            np_image, labels, boxes = session.run([image,  glabels, gbboxes])

            for j in range(labels.shape[0]):
                print('label=%d class(%s) at bbox[%f, %f, %f, %f]' % (
                    labels[j], labels[j], 
                    boxes[j][0], boxes[j][1],boxes[j][2],  boxes[j][3]))
                
                     
            bboxes_draw_on_img(np_image, labels, boxes, colors_plasma)
            cv2.imshow('Object Detection Image',np_image)
            cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)
test()
