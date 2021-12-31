import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from random import shuffle
import tensorflow as tf
import argparse

# Open annotation file
df = pd.read_csv('/kaggle/input/tensorflow-great-barrier-reef/train.csv')

df['path'] = None
for i in range(len(df)):
    df['path'].iloc[i] = f"../input/tensorflow-great-barrier-reef/train_images/video_{df['video_id'].iloc[i]}/{df['video_frame'].iloc[i]}.jpg"

def get_inp_shape(dataset):
    
    for i, (image, t_bbox, t_labels) in enumerate(dataset):
        input_shape = image.shape
        if i > 1: break
    return np.array(image.shape)
            


inp_shape = get_inp_shape(train_dataset)

def numpy_fc(idx, fc, outputs_types=(tf.float32, tf.float32, tf.int64), **params):
    """
    Call a numpy function on each given ID (`idx`) and load the associated image and labels (bbbox and cls)
    """
    def _np_function(_idx):
        return fc(_idx, **params)
    return tf.numpy_function(_np_function, [idx], outputs_types)


def pad_labels(images: tf.Tensor, t_bbox: tf.Tensor, t_class: tf.Tensor):
    """ Pad the bbox by adding [0, 0, 0, 0] at the end
    and one header to indicate how maby bbox are set.
    Do the same with the labels. 
    """
    nb_bbox = tf.shape(t_bbox)[0]

    bbox_header = tf.expand_dims(nb_bbox, axis=0)
    bbox_header = tf.expand_dims(bbox_header, axis=0)
    bbox_header = tf.pad(bbox_header, [[0, 0], [0, 3]])
    bbox_header = tf.cast(bbox_header, tf.float32)
    cls_header = tf.constant([[0]], dtype=tf.int64)

    # Padd bbox and class
    t_bbox = tf.pad(t_bbox, [[0, 20 - 1 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)
    t_class = tf.pad(t_class, [[0, 20 - 1 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)

    t_bbox = tf.concat([bbox_header, t_bbox], axis=0)
    t_class = tf.concat([cls_header, t_class], axis=0)

    return images, t_bbox, t_class


def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy

def load_data_index(index, df, scale):
    """
    returns xcyc_wh t_bbox
    """
    img = cv2.imread(df['path'].iloc[index], 1)
    ann = eval(df['annotations'].iloc[index])
    
    if ann:
        
        t_class = np.array([1 for obj in eval(df['annotations'].iloc[index])])
        bbox_list = np.array([np.array(list(obj.values())) for obj in eval(df['annotations'].iloc[index])])
        #convert from xy min wh to xymin xymax
        #t_bbox = np.concatenate([bbox_list[:, :2], bbox_list[:, :2] + bbox_list[:, 2:]], axis=-1)
        
       # convert from xy min wh to xy cent wh
        t_bbox = np.concatenate([((2*bbox_list[:, :2]) + bbox_list[:, 2:]) / 2, bbox_list[:, 2:]], axis=-1)
        
        
        
    else:
        t_class = np.array([0])
        t_bbox = np.expand_dims(np.array([0, 0, 0, 0]), axis=0)


    # resize img and bbox
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (width,height))
    t_bbox = t_bbox * scale
    
    # normalize bbox and image
    t_bbox = t_bbox / [width, height, width, height]
    resized_img = (resized_img / 255).astype(np.float32)
    # t_bbox from xymin_xymax to xyc_wh
    #t_bbox = np.concatenate([(bbox_list[:, :2] + bbox_list[:, 2:]) / 2, bbox_list[:, 2:] - bbox_list[:, :2]], axis=-1)

    # add augmentation here
    
    return resized_img.astype(np.float32), t_bbox.astype(np.float32), np.expand_dims(t_class, axis=-1)



img, tbbox, tclass = load_data_index(0, df, .5)
print(img.shape)
print(tbbox, tbbox.shape)
print(tclass,tclass.shape)

img, tbbox, tclass = load_data_index(16, df, .5)
print(img.shape)
print(tbbox, tbbox.shape)
print(tclass, tclass.shape)

img, tbbox, tclass = load_data_index(37, df, .5)
print(img.shape)
print(tbbox, tbbox.shape)
print(tclass, tclass.shape)


def load_dataset(df, batch_size, scale, epochs):
    
    indexes =[]
    for i in range(len(df['annotations'])):
        if eval(df['annotations'].iloc[i]) != []:
            indexes.append(i)
    shuffle(indexes)
    
    train_ind = indexes[:int(np.floor(0.8*len(indexes)))]
    val_ind = indexes[int(np.floor(0.8*len(indexes))):]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ind)
    train_dataset = train_dataset.map(lambda idx: numpy_fc(
        idx, load_data_index, 
        df=df, scale=scale)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    #train_dataset = train_dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    train_dataset = train_dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    train_dataset = train_dataset.repeat(epochs).batch(batch_size, drop_remainder=True) ## repeat num epochs **
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(val_ind)
    val_dataset = val_dataset.map(lambda idx: numpy_fc(
        idx, load_data_index, 
        df=df, scale=scale)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    #dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    val_dataset = val_dataset.map(pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    val_dataset = val_dataset.repeat(epochs).batch(batch_size, drop_remainder=True) ## repeat num epochs **
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


train_dataset, val_dataset = load_dataset(df, 8, .5, 50)


train_dataset, val_dataset = load_dataset(df, 8, .5)
for i, (image, t_bbox, t_labels) in enumerate(train_dataset):
    print("shapes", image.shape, t_bbox.shape, t_labels.shape)
    if i > 10: break
for i, (image, t_bbox, t_labels) in enumerate(val_dataset):
    print("shapes", image.shape, t_bbox.shape, t_labels.shape)
    if i > 10: break


