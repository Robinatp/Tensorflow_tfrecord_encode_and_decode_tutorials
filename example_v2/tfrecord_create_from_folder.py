#coding=utf-8
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io, transform

#将数据打包，转换成tfrecords格式，以便后续高效读取
def encode_to_tfrecords(file_dir,tfrecord_name,resize=None):
    '''
    此处我加载的数据目录如下：
    bluesky.0.jpg
    bluesky.0.jpg
    bluesky.0.jpg
    ....
    bluesky.1.jpg
    bluesky.1.jpg
    bluesky.1.jpg
    bluesky.1.jpg
    ...
    '''    
    writer=tf.python_io.TFRecordWriter(tfrecord_name)
    num_example=0
    for file in os.listdir(file_dir):
        f = file_dir + file
        image = io.imread(f)
#         plt.imshow(image)
#         plt.show()
        if resize is not None:
                image=cv2.resize(image,resize)
        height,width,nchannel=image.shape
        label=int(1 if(file.split('.')[0].find('bluesky')>0) else 0)
#         print f,label

        example=tf.train.Example(features=tf.train.Features(feature={
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
        serialized=example.SerializeToString()
        writer.write(serialized)
        num_example+=1
    print ("样本数据量：",num_example)
    writer.close()

#读取tfrecords文件
def decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    example=tf.parse_single_example(serialized,features={
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'nchannel':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    })
    label=tf.cast(example['label'], tf.int32)
    image=tf.decode_raw(example['image'],tf.uint8)
# Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
#     image = tf.cast(img, tf.float32) / 255 - 0.5
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image=tf.reshape(image,tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))
    
    print('decode_from_tfrecords: ',image)  
    print('decode_from_tfrecords: ',label)
    #label=example['label']
    return image,label

#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(image, label, batch_size,crop_size = 224):
    #数据扩充变换
    print('get_batch: ',image)  
    print('get_batch: ',label)
#     distorted_image = tf.reshape(image,[224,224,3])
    distorted_image = tf.image.resize_image_with_crop_or_pad(image, 228,228)#随机裁剪**************
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])#随机裁剪**************
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转***************s
#     distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
#     distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化
    distorted_image = tf.image.per_image_standardization(distorted_image)
    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
#     distorted_image = tf.cast(distorted_image, tf.float32)
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=16,capacity=1500,min_after_dequeue=1000)
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)

    # 调试显示
    #tf.image_summary('images', images)
    
    return images, tf.reshape(label_batch, [batch_size])

#这个是用于测试阶段，使用的get_batch函数
def get_test_batch(image, label, batch_size,crop_size):
    #数据扩充变换
    distorted_image=tf.image.central_crop(image,39./45.)
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])#随机裁剪
    images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)
    return images, tf.reshape(label_batch, [batch_size])


#测试上面的压缩、解压代码
def test():
#     encode_to_tfrecords(file_dir="picture/train/",tfrecord_name="picture/data/bluesky_train.tfrecords")
    image,label=decode_from_tfrecords('picture/data/bluesky_train.tfrecords')
    batch_image,batch_label=get_batch(image,label,5)#batch 生成测试
    init=tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for l in range(5):#每run一次，就会指向下一个样本，一直循环
            image_np,label_np=session.run([image,label])#每调用run一次，那么
            plt.imshow(image_np)
            plt.show()
            print(label_np)
            
            batch_image_np,batch_label_np=session.run([batch_image,batch_label])
            plt.imshow(batch_image_np[l,:,:,:])
            plt.show()
            print(batch_label_np[l])
#             print batch_image_np.shape
#             print batch_label_np.shape
    
        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)
# test()
