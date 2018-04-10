# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from flowers_data import *
from image_processing import *
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    
    dataset = FlowersData(subset="train")
#     images, labels = distorted_inputs(dataset)
    images, labels = inputs(dataset)

    
    print(images,labels)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()     
        threads= tf.train.start_queue_runners(coord=coord)
        
        for i in range(10):
            image_np,label_np=sess.run([images, labels])
            print(image_np[0,:,:,:])
            plt.imshow(image_np[0,:,:,:])
            plt.title('label name:'+str(label_np[0]))
            plt.show()
            
#             cv2.imshow('label name:',image_np[0,:,:,:])
#             cv2.waitKey(0)
            
        coord.request_stop()     
        coord.join(threads)