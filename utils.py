import tensorflow as tf
import numpy as np

from PIL import Image
from vgg19.vgg import Vgg19
from dataprocess import BatchDataProcess


class Utils(object):
    def __init__(self,
                 args):
        self.args = args

    def read_image(self, image_path):
        return np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    def get_sytleImg_features(self, image_path):
        data_process = BatchDataProcess(self.args)

        image = self.read_image(image_path)
        image = tf.expand_dims(image, 0)
        image = data_process.preprocess_image(image)

        vgg_style = Vgg19(self.args.vgg_path)
        vgg_style.build(image, clear_data=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            style_source = sess.run([vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1,
                                     vgg_style.conv4_1, vgg_style.conv5_1])

        return list(style_source)

    def get_matting_matrix_list(self,
                                matting_indices,
                                matting_values,
                                matting_shape):
        # First reshape the values to meet the function
        matting_values_reshape = tf.reshape(matting_values,
                                            [self.args.batch_size, -1])

        matting_matrix_list = []
        for idx in range(self.args.batch_size):
            matting_matrix = tf.SparseTensor(matting_indices[idx, :, :],
                                             matting_values_reshape[idx, :],
                                             matting_shape[idx, :])
            matting_matrix_list.append(matting_matrix)

        return matting_matrix_list


