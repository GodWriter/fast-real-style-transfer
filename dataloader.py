import os
import sys
import pickle
import numpy as np
import tensorflow as tf

import time
import math
import copy

from PIL import Image
from utils import getLaplacian


class DataSet(object):
    def __init__(self,
                 args):
        self.args = args

        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.affine_weight = 1e4

        self.image_list = os.listdir(self.args.image_path)

    def create_dataset(self):
        # get path of image and matrix
        image_list = os.listdir(self.args.image_path)
        matrix_list_length = len(os.listdir(self.args.matrix_path))

        # create the tf-record
        count = 0
        for image in image_list:
            mat_path = os.path.join(self.args.matrix_path, image.replace('jpg', 'pkl'))
            if os.path.exists(mat_path):
                save_path = os.path.join(self.args.dataSet, '%s.tfrecord' % image[:-4])

                writer = tf.python_io.TFRecordWriter(save_path)
                img = np.array(Image.open(os.path.join(self.args.image_path, image)).convert('RGB'), dtype=np.float32)
                with open(mat_path, 'rb') as f:
                    mat = pickle.load(f)

                features = {}

                # append the matrix
                features['indices'] = tf.train.Feature(int64_list=tf.train.Int64List(value=mat[0].reshape(-1)))
                features['indices-shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=mat[0].shape))
                features['values'] = tf.train.Feature(float_list=tf.train.FloatList(value=mat[1]))
                features['dense_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=mat[2]))

                # append the image
                features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))
                features['image_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape))

                tf_features = tf.train.Features(feature=features)
                tf_example = tf.train.Example(features=tf_features)
                tf_serialized = tf_example.SerializeToString()

                writer.write(tf_serialized)
                writer.close()

            if count % 10 == 0:
                print("Now the step is ", count, " and the total step is ", matrix_list_length)
            count += 1

        print("Creating has been done!")

    def parse_function(self, example_proto):
        disc = {'indices': tf.VarLenFeature(dtype=tf.int64),
                'indices-shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'values': tf.FixedLenFeature(shape=(1, 1240996), dtype=tf.float32),
                'dense_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'image_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}

        parsed_example = tf.parse_single_example(example_proto, disc)
        parsed_example['indices'] = tf.sparse.to_dense(parsed_example['indices'])
        parsed_example['image'] = tf.decode_raw(parsed_example['image'], tf.float32)

        parsed_example['indices'] = tf.reshape(parsed_example['indices'], parsed_example['indices-shape'])
        parsed_example['image'] = tf.reshape(parsed_example['image'], parsed_example['image_shape'])

        return parsed_example

    def load(self):
        data_list = []

        for data in os.listdir(self.args.dataSet):
            data_list.append(os.path.join(self.args.dataSet, data))

        dataset = tf.data.TFRecordDataset(data_list)
        new_dataset = dataset.map(self.parse_function)
        # shuffle_dataset = new_dataset.shuffle(buffer_size=len(data_list))
        batch_dataset = new_dataset.batch(self.args.batch_size)
        epoch_dataset = batch_dataset.repeat(self.args.num_epochs)

        iterator = epoch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        return next_element

    def test_dataset(self):
        dataset = DataSet(self.args)
        next_element = dataset.load()
        sess = tf.InteractiveSession()

        i = 1
        while True:
            try:
                indices, values, dense_shape, image = sess.run([next_element['indices'],
                                                                next_element['values'],
                                                                next_element['dense_shape'],
                                                                next_element['image']])
            except tf.errors.OutOfRangeError:
                print("End of  dataSet")
                break
            else:
                # 显示每个样本中所有feature信息，只显示scalar的值
                print('No.%d' % i)
                print('indices shape: %s | type: %s' % (indices.shape, indices.dtype))
                print('values shape: %s | type: %s' % (values.shape, values.dtype))
                print('dense_shape shape: %s | type: %s' % (dense_shape.shape, dense_shape.dtype))
                print('image shape: %s | type: %s' % (image.shape, image.dtype))
                print("=" * 50)
            i += 1

    def _add_to_tfrecord(self, filename, tfrecord_writer):
        content_image = np.array(Image.open(os.path.join(self.args.image_path, filename)).convert("RGB"),
                                 dtype=np.float32)

        # Get matting matrix
        matting = tf.to_float(getLaplacian(content_image / 255.))
        with tf.Session() as sess:
            mat = sess.run(matting)

        example = tf.train.Example(features=tf.train.Features(feature={
            'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=mat[0].reshape(-1))),
            'indices-shape': tf.train.Feature(int64_list=tf.train.Int64List(value=mat[0].shape)),
            'values': tf.train.Feature(float_list=tf.train.FloatList(value=mat[1])),
            'dense_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=mat[2])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content_image.tostring()])),
            'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=content_image.shape))
        }))

        tfrecord_writer.write(example.SerializeToString())

    def create_dataset_new(self):
        file_created = 0 # count the tf-record has been created
        file_saved = 0 # count the file has been saved

        while file_created < self.args.tfrecord_num:
            tf_filename = '%s/train_%03d.tfrecord' % (self.args.dataSet,
                                                      file_saved)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                file_created_per_record = 0
                while file_created < self.args.tfrecord_num and file_created_per_record < self.args.samples_per_file:
                    sys.stdout.write('\r>> Converting image %d/%d' % (file_created+1, self.TFRECORD_NUM))
                    sys.stdout.flush()
                    filename = self.image_list[file_created]
                    # img_name = filename[:-4]
                    self._add_to_tfrecord(filename, tfrecord_writer)
                    file_created += 1
                    file_created_per_record += 1
                file_saved += 1

        print('\nFinished converting to the tfrecord.')
