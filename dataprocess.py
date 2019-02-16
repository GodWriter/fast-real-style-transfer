import tensorflow as tf


class DataProcess(object):
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

    def preprocess_for_train(self, image):
        pass

    def preprocess_for_eval(self, image):
        pass

    def preprocess_image(self, image):
        if self.args.training:
            return self.preprocess_for_train(image)
        else:
            return self.preprocess_for_eval(image)

    def unprocess_image(self, image):
        pass
