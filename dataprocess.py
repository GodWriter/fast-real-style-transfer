import tensorflow as tf
import numpy as np


class BatchDataProcess(object):
    def __init__(self,
                 args):
        self.args = args

        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94

    def _mean_image_subtraction(self, image):
        # define the initial means, whose shape is [3]
        means = tf.constant([self._R_MEAN, self._G_MEAN, self._B_MEAN], dtype=tf.float32)
        # change the dims of means to [1, 1, ,1, 3]
        means_ = tf.expand_dims(tf.expand_dims(tf.expand_dims(means, 0), 0), 0)

        return image - means_

    def _mean_image_subtraction_without_sess(self, image):
        # define the initial means, whose shape is [3]
        means = np.array([self._R_MEAN, self._G_MEAN, self._B_MEAN], dtype=np.float32)
        # change the dims of means to [1, 1, ,1, 3]
        means_ = np.expand_dims(np.expand_dims(np.expand_dims(means, 0), 0), 0)

        return image - means_

    def preprocess_image(self, image):
        return self._mean_image_subtraction(image)

    def preprocess_image_without_sess(self, image):
        return self._mean_image_subtraction_without_sess(image)


class SingleDataProcess(object):
    def __init__(self,
                 args):
        """
        _RESIZE_SIDE_MIN: The lower bound for the smallest side of the image
                          for aspect-preserving resizing. If 'training' is 'False', then
                          this value is used for rescaling.
        _RESIZE_SIDE_MAX: The upper bound for the smallest side of the image
                          for aspect-preserving resizing. If 'training' is 'False', this
                          value is ignored. Otherwise, the resize side is sampled from
                          [_RESIZE_SIDE_MIN, _RESIZE_SIDE_MAX]
        """
        self.args = args

        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94

        self._RESIZE_SIDE_MIN = 256
        self._RESIZE_SIDE_MAX = 512

    def _smallest_size_at_least(self, image):
        # Get the origin image's shape
        source_shape = tf.shape(image)
        source_height, source_width = source_shape[0], source_shape[1]
        # Get the target image's shape
        target_height = tf.convert_to_tensor(self.args.output_height, dtype=tf.float32)
        target_width = tf.convert_to_tensor(self.args.output_width, dtype=tf.float32)

        scale = tf.cond(tf.greater(target_height/source_height, target_width/source_width),
                        lambda: target_height/source_height,
                        lambda: target_width/source_width)
        new_height = tf.to_int32(tf.round(source_height * scale))
        new_width = tf.to_int32(tf.round(source_width * scale))

        return new_height, new_width

    def aspect_preserving_resize(self, image):
        """
        Resizing the images preserving the original aspect ratio
        :param image: A 3-D image 'Tensor'
        :return: A 3-D tensor containing the resized image
        """
        new_height, new_width = self._smallest_size_at_least(image)

        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image,
                                                 [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)

        return resized_image

    def preprocess_for_train(self, image):
        pass

    def preprocess_for_eval(self, image):
        image = self.aspect_preserving_resize(image)

    def preprocess_image(self, image):
        if self.args.training:
            return self.preprocess_for_train(image)
        else:
            return self.preprocess_for_eval(image)

    def unprocess_image(self, image):
        pass
