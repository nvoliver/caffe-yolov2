# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:10:21 2016

@author: xingw
"""

import argparse
import caffe
import numpy as np


def main(args):
    model_filename = args.prototxt
    yoloweight_filename = args.weights
    caffemodel_filename = args.caffemodel

    print('model file is ' + model_filename)
    print('weights file is ' + yoloweight_filename)
    print('output caffemodel file is ' + caffemodel_filename)
    net = caffe.Net(model_filename, caffe.TEST)
    params = net.params.keys()

    # read weights from file and assign to the network
    net_weights_int = np.fromfile(yoloweight_filename, dtype=np.int32)
    trans_flag = (net_weights_int[0] > 1000 or net_weights_int[1] > 1000)
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    start = 4
    if (net_weights_int[0] * 10 + net_weights_int[1]) >= 2:
        start = 5

    print(trans_flag)

    net_weights_float = np.fromfile(yoloweight_filename, dtype=np.float32)
    # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
    net_weights = net_weights_float[start:]
    print(net_weights.shape)
    count = 0

    print("#Total Net Layers: %d" % len(net.layers))

    layercnt = 0
    for pr in params:
        layercnt = layercnt + 1
        lidx = list(net._layer_names).index(pr)
        layer = net.layers[lidx]
        if count == net_weights.shape[0] and (layer.type != 'BatchNorm'
                                              and layer.type != 'Scale'):
            print("WARNING: no weights left for %s" % pr)
            break
        if layer.type == 'Convolution':
            print(pr + "(conv)" + "-" + str(layercnt) + "-" +
                  str(len(net.params[pr]) > 1))
            # bias
            if len(net.params[pr]) > 1:
                bias_dim = net.params[pr][1].data.shape
            else:
                bias_dim = (net.params[pr][0].data.shape[0], )
            bias_size = np.prod(bias_dim)
            conv_bias = np.reshape(net_weights[count:count + bias_size],
                                   bias_dim)
            if len(net.params[pr]) > 1:
                assert bias_dim == net.params[pr][1].data.shape
                net.params[pr][1].data[...] = conv_bias
                conv_bias = None
            count = count + bias_size
            # batch_norm
            if lidx + 1 < len(
                    net.layers) and net.layers[lidx + 1].type == 'BatchNorm':
                bn_dims = (3, net.params[pr][0].data.shape[0])
                bn_size = np.prod(bn_dims)
                batch_norm = np.reshape(net_weights[count:count + bn_size],
                                        bn_dims)
                count = count + bn_size
            # weights
            dims = net.params[pr][0].data.shape
            weight_size = np.prod(dims)
            net.params[pr][0].data[...] = np.reshape(
                net_weights[count:count + weight_size], dims)
            count = count + weight_size

        elif layer.type == 'InnerProduct':
            print(pr + "(fc)")
            # bias
            bias_size = np.prod(net.params[pr][1].data.shape)
            net.params[pr][1].data[...] = np.reshape(
                net_weights[count:count + bias_size],
                net.params[pr][1].data.shape)
            count = count + bias_size
            # weights
            dims = net.params[pr][0].data.shape
            weight_size = np.prod(dims)
            if trans_flag:
                net.params[pr][0].data[...] = np.reshape(
                    net_weights[count:count + weight_size],
                    (dims[1], dims[0])).transpose()
            else:
                net.params[pr][0].data[...] = np.reshape(
                    net_weights[count:count + weight_size], dims)
            count = count + weight_size
        elif layer.type == 'BatchNorm':
            print(pr + "(batchnorm)")
            net.params[pr][0].data[...] = batch_norm[1]  # mean
            net.params[pr][1].data[...] = batch_norm[2]  # variance
            net.params[pr][2].data[...] = 1.0  # scale factor
        elif layer.type == 'Scale':
            print(pr + "(scale)")
            if batch_norm is not None:
                net.params[pr][0].data[...] = batch_norm[0]  # scale
            batch_norm = None
            if len(net.params[pr]) > 1:
                net.params[pr][1].data[...] = conv_bias  # bias
                conv_bias = None
        else:
            print("WARNING: unsupported layer, " + pr)
    if np.prod(net_weights.shape) != count:
        print("ERROR: size mismatch: %d" % count)
    net.save(caffemodel_filename)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Convert YOLO weights to Caffe weights')
    argparser.add_argument(
        '-m',
        '--prototxt',
        type=str,
        required=True,
        help='Caffe prototxt input')
    argparser.add_argument(
        '-w', '--weights', type=str, required=True, help='YOLO weights input')
    argparser.add_argument(
        '-o',
        '--caffemodel',
        type=str,
        required=True,
        help='Caffe weights output')
    args = argparser.parse_args()
    main(args)
