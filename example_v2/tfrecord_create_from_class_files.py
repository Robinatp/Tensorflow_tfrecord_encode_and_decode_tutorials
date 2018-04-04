#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
import cv2
from scipy import misc

cwd = os.getcwd()

def create_tfrecord(filename):
    '''
    此处我加载的数据目录如下：
    0 -- img1.jpg
         img2.jpg
         img3.jpg
         ...
    1 -- img1.jpg
         img2.jpg
         ...
    2 -- ...
    ...
    '''
    num_classes={'Ok','Faile'}#不同分类文件夹名, Ok-0,Faile-1
    num_example=0
    root = 'picture/'
    writer = tf.python_io.TFRecordWriter(filename)
    for index, name in enumerate(num_classes):
        class_path = root + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            print(img_path)
#             img = Image.open(img_path)
#             img = img.resize((224, 224))
            img = io.imread(img_path)
            img=cv2.resize(img,(224,224))
#             plt.imshow(img)
#             plt.title('file name:'+img_path+"  "+str(num_example))
#             plt.show()

            img_raw = img.tobytes() #将图片转化为原生bytes,方便存储在tfrecord中
#             image = io.imread(img_path)
#             image=cv2.resize(image,(224,224))

            example = tf.train.Example(features=tf.train.Features(
                feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
            num_example+=1
    writer.close()
    print "样本数据量：",num_example

def read_and_decode_from_tfrecord(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])#[W,H,C]
# Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
#     image = tf.cast(img, tf.float32) / 255 - 0.5
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(image, label, batch_size,crop_size = 224):
    #数据扩充变换
    print('get_batch: ',image)  
    print('get_batch: ',label)
    #distorted_image = tf.reshape(image,[224,224,3])
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
#     distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
#     distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化
    distorted_image = tf.image.per_image_standardization(distorted_image)
    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=16,capacity=1500,min_after_dequeue=1000)
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)
#     images = tf.cast(images, tf.float32)
    # 调试显示
    #tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def test():
#     create_tfrecord("picture/data/bluesky_valia.tfrecords")# create tfrecode
    img, label = read_and_decode_from_tfrecord("picture/data/bluesky_valia.tfrecords")#parse the tfrecode

#     img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                     batch_size=30, 
#                                                     capacity=2000,
#                                                     min_after_dequeue=1000)
    img_batch, label_batch =get_batch(img,label,30)
    #初始化所有的op
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        #启动队列
        coord=tf.train.Coordinator()     
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            image_np,label_np=sess.run([img,label])#每调用run一次，那么
            
            plt.imshow(image_np)
            plt.title('label name:'+str(label_np))
            plt.show()
            
#             misc.imsave('picture/valia/'+str(i)+'_''Label_'+str(label_np)+'.jpg', image_np)
            
            val, l= sess.run([img_batch, label_batch])
# #             l = to_categorical(l, 12)
#             print(val.shape, l[i])
            plt.imshow(val[i,:,:,:])
            plt.show()
        coord.request_stop()     
        coord.join(threads)
# test()

