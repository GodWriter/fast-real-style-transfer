import tensorflow as tf


class Layers(object):
    @staticmethod
    def conv2d(x, in_channels, out_channels, kernel,
               strides, mode='REFLECT', name='conv'):
        with tf.variable_scope(name):
            # create the filters
            w_shape = [kernel, kernel, in_channels, out_channels]
            weight = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name='weight')
            # pad the image
            pad_num = int(kernel / 2)
            x_padded = tf.pad(x, [[0, 0], [pad_num, pad_num], [pad_num, pad_num], [0, 0]], mode=mode)

            return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID')

    @staticmethod
    def conv2d_transpose(x, in_channels, out_channels,
                         kernel, strides, name):
        with tf.variable_scope(name):
            # create the filters
            w_shape = [kernel, kernel, out_channels, in_channels]
            weight = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name='weight')
            # get the shape, when padding=SAME, input = s*output
            batch_size = tf.shape(x)[0]
            height, width = tf.shape(x)[1]*strides, tf.shape(x)[2]*strides
            output_shape = tf.stack([batch_size, height, width, out_channels])

            return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], padding="SAME")

    @staticmethod
    def resize_conv2d(x, in_channels, out_channels, kernel,
                      strides, training, name):
        with tf.variable_scope(name):
            # get the origin shape
            height = x.get_shape()[1].value if training else tf.shape(x)[1]
            width = x.get_shape()[2].value if training else tf.shape(x)[2]
            # compute the output shape
            new_height, new_width = height*strides*strides, width*strides*strides

            # Get the temporal image
            x_resized = tf.image.resize_images(x, [new_height, new_width],
                                               tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return Layers.conv2d(x_resized, in_channels, out_channels, kernel, strides)

    @staticmethod
    def instance_norm(x):
        epsilon = 1e-9
        # get each channel's mean and var
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

        # there should be matrix compute with  matrix
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    @staticmethod
    def batch_norm(x, size, training, decay=0.999):
        pass

    @staticmethod
    def relu(x):
        relu = tf.nn.relu(x)
        # convert nan to zero (nan != nan)
        nan_to_zero = tf.where(tf.equal(relu, relu), relu, tf.zeros_like(relu))

        return nan_to_zero

    @staticmethod
    def residual(x, filters, kernel, strides, name):
        with tf.variable_scope(name):
            conv1 = Layers.conv2d(x, filters, filters, kernel, strides)
            relu_ = Layers.relu(conv1)
            conv2 = Layers.conv2d(relu_, filters, filters, kernel, strides)

            residual = x + conv2

        return residual
