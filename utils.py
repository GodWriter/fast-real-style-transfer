import tensorflow as tf
import numpy as np

import scipy.ndimage as spi
import scipy.sparse as sps

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

    def compute_matting_matrix(self, input_image):
        # the numpy matting matrix
        _indices = np.zeros((self.args.batch_size, 1240996, 2))
        _values = np.zeros((self.args.batch_size, 1, 1240996))
        _shape = np.zeros((self.args.batch_size, 2))

        for idx in range(self.args.batch_size):
            mat = getLaplacian(input_image[idx, :, :, :] / 255.)
            _indices[idx, :, :] = mat[0]
            _values[idx, :, :] = mat[1]
            _shape[idx, :] = mat[2]

        return _indices, _values, _shape


def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse


def getLaplacian(img):
    h, w, _ = img.shape
    coo = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return indices, coo.data, coo.shape




