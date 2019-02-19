import argparse


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "TensorFlow implementation of fast-style-GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--module', type=str, default='test',
                        help='Module to select: train, test, test_dataset, create_dataset, train_without_affine')
    parser.add_argument('--training', type=bool, default=True,
                        help='If the model is train, this argument should be true, else False')
    parser.add_argument('--GPU', type=str, default='0',
                        help='GPU used to train the model')
    parser.add_argument('--image_path', type=str, default='data/images',
                        help='Path of the image data')
    parser.add_argument('--matrix_path', type=str, default='data/matrix',
                        help='Path of the matrix data')
    parser.add_argument('--dataSet', type=str, default='data/tf-record',
                        help='Path of the tf-record dataSet')
    parser.add_argument('--vgg_path', type=str, default='vgg19/vgg19.npy',
                        help='Path of the trained vgg19 model')
    parser.add_argument('--style_image_path', type=str, default='data/style-image/starry.jpg',
                        help='Path of the style image')
    parser.add_argument('--output_height', type=int, default=224,
                        help='The height of the unprocessed image after processed')
    parser.add_argument('--output_width', type=int, default=224,
                        help='The width of the unprocessed image after processed')
    parser.add_argument('--content_weight', type=float, default=1,
                        help='Weight of content loss when computing the total loss')
    parser.add_argument('--style_weight', type=float, default=1,
                        help='Weight of style loss when computing the total loss')
    parser.add_argument('--tv_weight', type=float, default=1,
                        help='Weight of total variation loss when computing the total loss')
    parser.add_argument('--affine_weight', type=float, default=1,
                        help='Weight of affine loss when computing the total loss')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size of the data')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of the train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-03,
                        help='Learning rate for the optimizer')
    parser.add_argument('--log_path', type=str, default='log',
                        help='Path to save the log')
    parser.add_argument('--model_path', type=str, default='model',
                        help='Path to save the model')
    parser.add_argument('--save_epoch', type=int, default=1000,
                        help='The frequency to save the model')
    parser.add_argument('--save_summary', type=int, default=10,
                        help='The frequency to save the summary')
    parser.add_argument('--model_file', type=str, default='model/fast-real-model-done.ckpt',
                        help='The model used to test')
    parser.add_argument('--test_image', type=str, default='test_image.jpg',
                        help='File path of the test image')

    return parser.parse_args()
