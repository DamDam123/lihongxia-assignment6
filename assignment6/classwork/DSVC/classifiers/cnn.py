from builtins import object
import numpy as np

from DSVC.layers import *
from DSVC.fast_layers import *
from DSVC.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        # 卷积层
        self.params['W1'] = np.random.randn(num_filters,C,filter_size,filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        # 全连接层
        self.params['W2'] = np.random.randn(int(num_filters * H * W/4),hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        # 全连接层用于最后分类
        self.params['W3'] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        # self.params['gamma'] = np.ones(num_filters)
        # self.params['beta'] = np.zeros(num_filters)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # 池化层到全连接层，要将数据转为（N,F * nHp(池化层之后的新高) * nWp(池化层之后的新宽)），而经过池化层之后高度和宽度都变为原始数据的一半
        # F * H * W / 4 == F * nHp * nWp
        # 所以W2的维度为（F * H * W / 4 , hidden_dim）
        # a1也要转为(N, F * nHp * nWp)
        a1_reshape = a1.reshape(a1.shape[0], -1)
        a2, cache2 = affine_relu_forward(a1_reshape, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss_without_reg, dscores = softmax_loss(scores, y)
        # loss加L2正则化
        loss = loss_without_reg + 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        da2, dW3, db3 = affine_backward(dscores,cache3)
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3

        da1, dW2, db2 = affine_relu_backward(da2, cache2)
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2

        da1_reshape = da1.reshape(*a1.shape)

        dx, dW1, db1 = conv_relu_pool_backward(da1_reshape, cache1)
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] =db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
