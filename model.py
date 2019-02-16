import tensorflow as tf

from layers import Layers


class StyleGenerator(object):
    def __init__(self,
                 args):
        self.args = args

    @staticmethod
    def _pad(image):
        return tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    @staticmethod
    def _remove_pad(y):
        height, width = tf.shape(y)[1], tf.shape(y)[2]
        y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

        return y

    def model(self, image, training=False):
        # Less border effect
        image = self._pad(image)

        with tf.variable_scope('styleGenerator'):
            # down sample
            conv1 = Layers.conv2d(image, 3, 32, 9, 1, name='conv1')
            norm1 = Layers.instance_norm(conv1)
            relu1 = Layers.relu(norm1)

            conv2 = Layers.conv2d(relu1, 32, 64, 3, 2, name='conv2')
            norm2 = Layers.instance_norm(conv2)
            relu2 = Layers.relu(norm2)

            conv3 = Layers.conv2d(relu2, 64, 128, 3, 2, name='conv3')
            norm3 = Layers.instance_norm(conv3)
            relu3 = Layers.relu(norm3)

            # residual network
            res1 = Layers.residual(relu3, 128, 3, 1, name='res1')
            res2 = Layers.residual(res1, 128, 3, 1, name='res2')
            res3 = Layers.residual(res2, 128, 3, 1, name='res3')
            res4 = Layers.residual(res3, 128, 3, 1, name='res4')
            res5 = Layers.residual(res4, 128, 3, 1, name='res5')

            # up sample
            deconv1 = Layers.resize_conv2d(res5, 128, 64, 3, 2, name='deconv1', training=training)
            denorm1 = Layers.instance_norm(deconv1)
            derelu1 = Layers.relu(denorm1)

            deconv2 = Layers.resize_conv2d(derelu1, 64, 32, 3, 2, name='deconv2', training=training)
            denorm2 = Layers.instance_norm(deconv2)
            derelu1 = Layers.relu(denorm2)

            deconv3 = Layers.resize_conv2d(derelu1, 32, 3, 9, 1, name='deconv3', training=training)
            denorm3 = Layers.instance_norm(deconv3)
            detanh3 = tf.nn.tanh(denorm3)

            y = (detanh3 + 1) * 127.5

            # Remove border effect
            y = self._remove_pad(y)

            return y







