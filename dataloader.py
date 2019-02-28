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

        self.image_list = os.listdir(self.args.image_path)

    def parse_function(self, example_proto):
        disc = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'image_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}

        parsed_example = tf.parse_single_example(example_proto, disc)

        parsed_example['image'] = tf.decode_raw(parsed_example['image'], tf.float32)
        parsed_example['image'] = tf.reshape(parsed_example['image'], parsed_example['image_shape'])

        return parsed_example

    def load(self):
        data_list = []

        for data in os.listdir(self.args.dataSet):
            data_list.append(os.path.join(self.args.dataSet, data))

        dataset = tf.data.TFRecordDataset(data_list)
        new_dataset = dataset.map(self.parse_function)
        shuffle_dataset = new_dataset.shuffle(buffer_size=len(data_list))
        batch_dataset = shuffle_dataset.batch(self.args.batch_size)
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
                image = sess.run(next_element['image'])
            except tf.errors.OutOfRangeError:
                print("End of  dataSet")
                break
            else:
                # 显示每个样本中所有feature信息，只显示scalar的值
                print('No.%d' % i)
                print('image shape: %s | type: %s' % (image.shape, image.dtype))
                print("=" * 50)
            i += 1

    def _add_to_tfrecord(self, filename, tfrecord_writer):
        content_image = np.array(Image.open(os.path.join(self.args.image_path, filename)).convert("RGB"),
                                 dtype=np.float32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content_image.tostring()])),
            'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=content_image.shape))
        }))

        tfrecord_writer.write(example.SerializeToString())

    def create_dataset(self):
        file_created = 0 # count the tf-record has been created
        file_saved = 0 # count the file has been saved

        while file_created < self.args.tfrecord_num:
            tf_filename = '%s/train_%03d.tfrecord' % (self.args.dataSet,
                                                      file_saved)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                file_created_per_record = 0
                while file_created < self.args.tfrecord_num and file_created_per_record < self.args.samples_per_file:
                    sys.stdout.write('\r>> Converting image %d/%d' % (file_created, self.args.tfrecord_num))
                    sys.stdout.flush()
                    filename = self.image_list[file_created]
                    # img_name = filename[:-4]
                    self._add_to_tfrecord(filename, tfrecord_writer)
                    file_created += 1
                    file_created_per_record += 1
                file_saved += 1

        print('\nFinished converting to the tfrecord.')
