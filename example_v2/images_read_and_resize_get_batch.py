# -*- coding: GBK -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from scipy import misc
import glob
import cv2
import Image
w = 224
h = 224
c = 3

def images_resize_and_return_list(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    blueskys = []
    label_blueskys = []
    others = []
    label_others = []
    for file in os.listdir(file_dir):
        #name = file.split(sep='.')
        if file =='Ok':
            for f1 in os.listdir(file_dir + file):
                #print(file_dir + file +'/'+ f1)
                blueskys.append(file_dir + file +'/'+ f1)
                label_blueskys.append(0)
                img = io.imread(file_dir + file +'/'+ f1)
                img = transform.resize(img, (w, h, c))
                #plt.imshow(img)
                #plt.show()
                misc.imsave('picture/train/bluesky.'+str(len(blueskys))+'.jpg', img)
                print('picture/train/bluesky.'+str(len(blueskys))+'.jpg'+"  0")
        if file =='Faile':
            for f2 in os.listdir(file_dir + file):
                #print(file_dir + file+'/'+f2)
                others.append(file_dir + file+'/'+f2)
                label_others.append(1)
                img = io.imread(file_dir + file +'/'+ f2)
                img = transform.resize(img, (w, h, c))
                #plt.imshow(img)
                #plt.show()
                misc.imsave('picture/train/nobluesky.'+str(len(others))+'.jpg', img)
                print('picture/train/nobluesky.'+str(len(others))+'.jpg'+"  1")
    print('There are %d bluesky\nThere are %d other' %(len(blueskys), len(others)))
    
    image_list = np.hstack((blueskys, others))
    label_list = np.hstack((label_blueskys, label_others))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    print image_list
    print label_list
    return image_list, label_list

def get_batch_for_train_and_valia(data, label, ratio): 
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val   = data[s:]
    y_val   = label[s:]
    return x_train,y_train,x_val,y_val

def images_resize_and_return_tensor(read_path,save_path=None):
    '''
    Args:
        read_path: file directory for read,resize the images
        save_path: file directory for save,save the resize the image
    Returns:
        tensor of images and labels
    '''
    cate   = [read_path + x for x in os.listdir(read_path) if os.path.isdir(read_path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
#             img = io.imread(im)
#             img=cv2.resize(img,(224,224))
#             img = tf.reshape(img, [224, 224, 3])#[W,H,C]
            distorted_image = tf.image.per_image_standardization(img)
#             image_std = tf.cast(img, tf.float32) * (1. / 255) - 0.5
            if save_path is not None:
                misc.imsave(save_path+str(len(labels))+'_Label_'+str(idx)+'.jpg', img)
            imgs.append(distorted_image)
            labels.append(idx)
    print('the toal pictures is',len(labels))
    return  tf.reshape(imgs, [len(imgs), 224, 224, 3]),  np.asarray(labels, np.int32) #np.asarray(imgs, np.float32), np.asarray(labels, np.int32)




def images_read_from_folder(file_dir):
    imgs = []
    for file in os.listdir(file_dir):
#         print(file_dir + file)
        img = io.imread(file_dir + file )
#         img = transform.resize(img, (w, h, c))
        imgs.append(img)

#         plt.imshow(img)
#         plt.show()

    images = np.asarray(imgs, np.float32)
    return images


#######################################function test for demo###########################################
#1
# image_list, label_list = images_resize_and_return_list('picture/')
#2
# image_list,label_list = images_resize_and_return_tensor('picture/test/',"picture/valia/")
# print(image_list.shape)
# print(label_list)
#3
# x_train,y_train,x_val,y_val = get_batch_for_train_and_valia(image_list,label_list ,0.8)
# print(x_train.shape)
# print(y_train)
# print(x_val.shape)
# print(y_val)
#4
# images = images_read_from_folder('picture/valia/')
# print(images.shape)




    
