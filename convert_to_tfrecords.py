import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import skimage
import skimage.io


# Source: https://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow
# Note: modified from source
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, output_directory, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = 1

    filename = os.path.join(output_directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def read_image(file_name, images_path):
    image = skimage.io.imread(images_path + file_name)
    return image

def extract_image_index_make_label(img_name):
    remove_ext = img_name.split(".")[0]
    name, serie, repetition, char = remove_ext.split("_")
    label = int(char) + 1000 * int(repetition) + 1000_000 * int(serie)
    return label

images_path = "/kaggle/input/chinese-mnist/data/data/"
image_list = os.listdir(images_path)
images = []
labels = []
for img_name in tqdm(image_list):
    images.append(read_image(img_name, images_path))
    labels.append(extract_image_index_make_label(img_name))
images_array = np.array(images)
labels = np.array(labels)
print(images_array.shape, labels.shape)

convert_to(images_array, labels, ".", "chinese_mnist")