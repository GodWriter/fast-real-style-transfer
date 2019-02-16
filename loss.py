import tensorflow as tf


class Loss(object):
    def __init__(self,
                 args):
        self.args = args

    def gram_matrix(self, layer):
        shape = tf.shape(layer)
        image_num, image_channels = shape[0], shape[3]
        image_width, image_height = shape[1], shape[2]

        channel_vectors = tf.reshape(layer, tf.stack([image_num, -1, image_channels]))
        image_grams = tf.matmul(channel_vectors, channel_vectors, transpose_a=True) / \
                      tf.to_float(image_width * image_height * image_channels)

        return image_grams

    def content_loss(self, content_source, content_generate):
        return tf.reduce_mean(tf.squared_difference(content_source, content_generate)) \
               * self.args.content_weight

    def style_loss(self, style_source, style_generate):
        style_loss = 0

        for source, generated in zip(style_source, style_generate):
            size = tf.size(source)
            layer_style_loss = tf.nn.l2_loss(self.gram_matrix(source) -
                                             self.gram_matrix(generated)) * 2 / tf.to_float(size)
            style_loss += layer_style_loss
        style_loss = style_loss * self.args.style_weight

        return style_loss

    def total_variation_loss(self, generate_image):
        shape = tf.shape(generate_image)
        image_height, image_width = shape[1], shape[2]

        y = tf.slice(generate_image, [0, 0, 0, 0], tf.stack([-1, image_height - 1, -1, -1])) - \
            tf.slice(generate_image, [0, 1, 0, 0], [-1, -1, -1, -1])
        x = tf.slice(generate_image, [0, 0, 0, 0], tf.stack([-1, -1, image_width - 1, -1])) - \
            tf.slice(generate_image, [0, 0, 1, 0], [-1, -1, -1, -1])
        tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + \
                  tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
        tv_loss = tv_loss * self.args.tv_weight

        return tv_loss

    def affine_loss(self, generated_image, matting_matrix_list):
        loss_affine = 0.0
        generated_image = generated_image / 255.

        for idx in range(self.args.batch_size):
            for vc in tf.unstack(generated_image[idx, :, :, :], axis=-1):
                vc_ravel = tf.reshape(tf.transpose(vc), [-1])
                loss_affine += tf.matmul(tf.expand_dims(vc_ravel, 0), tf.sparse_tensor_dense_matmul(
                                         matting_matrix_list[idx], tf.expand_dims(vc_ravel, -1)))
        loss_affine = loss_affine * self.args.affine_weight

        return loss_affine

    def print_loss(self, loss_content, loss_style, loss_tv, loss_affine):
        print('loss_content: ', loss_content, 'loss_style: ', loss_style,
              'loss_tv: ', loss_tv, 'loss_affine: ', loss_affine)

