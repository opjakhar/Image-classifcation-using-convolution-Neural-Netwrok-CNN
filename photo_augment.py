
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
train_data=pd.read_csv("train.csv")
train_labels=np.array(train_data['label'])
train_data_features = np.array(train_data.iloc[:,1:785])
train_data=train_data_features
#train_data=train_data[0:100,:]
def photo_augmentation(photo1):
    photo_rotation = []
    photo_rotation1 = []
    photo_rotation2 = []
    photo_dimension =28
    tf.reset_default_graph()
    pl1 = tf.placeholder(tf.float32, shape = (photo_dimension, photo_dimension, 1))
    k = tf.placeholder(tf.int32)
    tf_photo = tf.image.rot90(pl1, k = k)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for pic_number in photo1:
            for i in range(1):
                photo_img = session.run(tf_photo, feed_dict = {pl1: pic_number, k:1})
                photo_img1 = session.run(tf_photo, feed_dict = {pl1: pic_number, k:2})
                photo_img2 = session.run(tf_photo, feed_dict = {pl1: pic_number, k:3})
                photo_rotation.append(photo_img)
                photo_rotation1.append(photo_img1)
                photo_rotation2.append(photo_img2)

    photo_rotation = np.array(photo_rotation, dtype = np.float32)
    photo_rotation1 = np.array(photo_rotation1, dtype = np.float32)
    photo_rotation2 = np.array(photo_rotation2, dtype = np.float32)
    return photo_rotation,photo_rotation1,photo_rotation2

batch_size=1
label_count=0;
for i in range(train_data.shape[0]):
    print("augmenting image {}".format(i))
    photo1=train_data[i,:]
    photo1 = np.array(photo1).reshape(1,28,28,1)
#   plt.imshow(np.squeeze(photo1))
#   plt.show()
#   print(phot1.shape)
#   plt.imshow(photo1)
#   plt.show()
    rotated_pic,rotated_pic1,rotated_pic2 =photo_augmentation(photo1)
    #print(rotated_pic.shape)
    append=np.squeeze(rotated_pic).reshape(1,784)
    append1=np.squeeze(rotated_pic1).reshape(1,784)
    append2=np.squeeze(rotated_pic2).reshape(1,784)
    if label_count==0 and i==0:
        final_append=np.vstack((append,append1,append2))
        dummy_lables=train_labels[i]
        dummy_lables=np.append(dummy_lables,train_labels[i])
        dummy_lables=np.append(dummy_lables,train_labels[i])
        label_count=label_count+1
    else:
        final_append=np.vstack((final_append,append,append1,append2))
        dummy_lables=np.append(dummy_lables,train_labels[label_count])
        dummy_lables=np.append(dummy_lables,train_labels[label_count])
        dummy_lables=np.append(dummy_lables,train_labels[label_count])
        label_count=label_count+1
    #-----printing  augmented pictures
#     print(np.squeeze(rotated_pic).shape)
#     plt.imshow(np.squeeze(rotated_pic))
#     plt.show()
#     plt.imshow(np.squeeze(rotated_pic1))
#     plt.show()
#     plt.imshow(np.squeeze(rotated_pic2))
#     plt.show()
    
aug_train_data=np.vstack((train_data,final_append))
aug_train_labels=np.append(train_labels,dummy_lables)
print(aug_train_data.shape)
print(aug_train_labels.shape)


# In[15]:


import numpy as np
import os
import cv2

# image : ndarray
#     Input image data. Will be converted to float.
# mode : str
#     One of the following strings, selecting the type of noise to add:

#     'gauss'     Gaussian-distributed additive noise.
#     'poisson'   Poisson-distributed noise generated from the data.
#     's&p'       Replaces random pixels with 0 or 1.
#     'speckle'   Multiplicative noise using out = image + n*image,where
#                 n is uniform noise with specified mean & variance.

train_data=pd.read_csv("train.csv")
train_labels=np.array(train_data['label'])
train_data_features = np.array(train_data.iloc[:,1:785])
train_data=train_data_features
#train_data=train_data[0:100,:]
image =train_data[0,:]

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,1)
        gauss = gauss.reshape(row,col,1)        
        noisy = image + image * gauss
        return noisy
noise_typ='poission'
image=noisy(noise_typ,image)
plt.imshow(image)
plt.show()


# In[13]:


image.type

