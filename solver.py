import os
import time
import tensorflow as tf

from dataloader import DataSet
from dataprocess import BatchDataProcess
from model import StyleGenerator
from vgg19.vgg import Vgg19
from loss import Loss
from utils import Utils


class Solver(object):
    def __init__(self, args):
        self.args = args

    def train(self):
        utils = Utils(self.args)
        data_preprocess = BatchDataProcess(self.args)

        # create the dataSet
        dataset = DataSet(self.args)
        next_element = dataset.load()

        # create the input for the image
        input_image = tf.placeholder(dtype=tf.float32, shape=[self.args.batch_size, 224, 224, 3])

        # create the input for the matting matrix
        matting_indices = tf.placeholder(dtype=tf.int64, shape=[self.args.batch_size, 1240996, 2])
        matting_values = tf.placeholder(dtype=tf.float32, shape=[self.args.batch_size, 1, 1240996])
        matting_shape = tf.placeholder(dtype=tf.int64, shape=[self.args.batch_size, 2])

        # create the generate model
        style_net = StyleGenerator(self.args)
        generate_image = style_net.model(input_image, training=self.args.training)

        # create the Vgg19
        vgg_source = Vgg19(self.args.vgg_path)
        vgg_generate = Vgg19(self.args.vgg_path)

        # There has a problem whether the image is rgb or bgr
        vgg_source.build(input_image, clear_data=False)
        vgg_generate.build(generate_image, clear_data=False)

        # Get the features for computing the content loss
        content_source = vgg_source.conv4_2
        content_generate = vgg_generate.conv4_2

        style_source = utils.get_sytleImg_features(self.args.style_image_path)
        style_generate = [vgg_generate.conv1_1, vgg_generate.conv2_1, vgg_generate.conv3_1,
                          vgg_generate.conv4_1, vgg_generate.conv5_1]

        loss = Loss(self.args)
        # Computing the ordinary loss
        loss_content = loss.content_loss(content_source, content_generate)
        loss_style = loss.style_loss(style_source, style_generate)
        loss_tv = loss.total_variation_loss(generate_image)

        # Computing the affine loss
        matting_matrix_list = utils.get_matting_matrix_list(matting_indices, matting_values, matting_shape)
        loss_affine = loss.affine_loss(generate_image, matting_matrix_list)

        # Computing the total loss with or without affine loss
        total_loss_without_affine = loss_content + loss_style + loss_tv
        total_loss_with_affine = total_loss_without_affine + loss_affine

        # Add summary
        tf.summary.scalar('losses/content_loss', loss_content)
        tf.summary.scalar('losses/style_loss', loss_style)
        tf.summary.scalar('losses/total_variable_loss', loss_tv)
        tf.summary.scalar('losses/affine_loss', loss_affine[0, 0])
        tf.summary.scalar('total_loss', total_loss_with_affine[0, 0])

        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.args.log_path)

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1e-08)
        train_op = optimizer.minimize(total_loss_with_affine,
                                      var_list=tf.trainable_variables(scope='styleGenerator'))

        # Save the model
        saver = tf.train.Saver(var_list=tf.trainable_variables(scope='styleGenerator'),
                               write_version=tf.train.SaverDef.V1)

        # initial the global parameters
        init_global = tf.global_variables_initializer()

        step = 0
        with tf.Session() as sess:
            sess.run(init_global)

            while True:
                try:
                    # First get the data
                    indices, values, shape, image = sess.run([next_element['indices'],
                                                              next_element['values'],
                                                              next_element['dense_shape'],
                                                              next_element['image']])
                    # Then preprocess the data
                    image = data_preprocess.preprocess_image_without_sess(image)
                    # Finally run the train_op
                    sess.run(fetches=[train_op], feed_dict={input_image: image,
                                                            matting_indices: indices,
                                                            matting_values: values,
                                                            matting_shape: shape})

                    # save and print the summary
                    if step % self.args.save_summary == 0:
                        _summary, _loss_content, _loss_style, _loss_tv, _loss_affine = sess.run([summary, loss_content,
                                                                                                 loss_style, loss_tv,
                                                                                                 loss_affine],
                                                                                                 {input_image: image,
                                                                                                  matting_indices: indices,
                                                                                                  matting_values: values,
                                                                                                  matting_shape: shape})
                        writer.add_summary(_summary, step)
                        print('Step: ', step, ' loss_content: ', _loss_content, ' loss_style: ', _loss_style,
                              ' loss_tv: ', _loss_tv, ' loss_affine: ', _loss_affine)
                        writer.flush()

                    # save the model
                    if step % self.args.save_epoch == 0:
                        saver.save(sess,
                                   os.path.join(self.args.model_path, 'fast-real-model-%s.ckpt' % step))
                except tf.errors.OutOfRangeError:
                    print("End of training!")
                    saver.save(sess,
                               os.path.join(self.args.model_path, 'fast-real-model.ckpt-done'))
                    break
                else:
                    pass
                step += 1

    def test(self):
        style_model = StyleGenerator(self.args)
        utils = Utils(self.args)
        data_preprocess = BatchDataProcess(self.args)

        image = utils.read_image(self.args.test_image)
        image = tf.expand_dims(image, 0)
        image = data_preprocess.preprocess_image(image)

        generated = style_model.model(image, training=self.args.training)
        generated = tf.squeeze(generated, [0])
        generated = tf.image.convert_image_dtype(generated, dtype=tf.uint8)

        saver = tf.train.Saver(tf.global_variables(),
                               write_version=tf.train.SaverDef.V1)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.args.model_file)

            # Make sure 'generated' directory exists.
            generated_path = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            with open(generated_path, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                print('Elapsed time: %fs' % (end_time - start_time))



