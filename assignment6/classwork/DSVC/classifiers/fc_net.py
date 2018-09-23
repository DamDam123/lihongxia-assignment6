from builtins import range
from builtins import object
import numpy as np

from DSVC.layers import *
from DSVC.layer_utils import *

from DSVC.layers import affine_backward, relu_backward, affine_forward, relu_forward


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # np.random.normal(loc,scale,size)正态分布
        # loc表示概率分布的均值（期望）
        # scale表示概率分布的标准差
        # size表示输出的形状，参数为int or tuple of ints， If the given shape is, e.g., (m, n, k)
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    # 有正则化的loss
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        # 网络结构：affine - relu - affine - softmax

        # 用layers.py写
        # #cache1包括X, W1, b1
        # hidden_layer_one, cache1 = affine_forward(X, W1, b1)
        # #cache2包括hidden_layer_one
        # hidden_layer_two, cache2 = relu_forward(hidden_layer_one)
        # #cache包括hidden_layer_two, W2, b2
        # scores, cache3 = affine_forward(hidden_layer_two, W2, b2)

        # 用layer_utils.py写
        affine_relu_out, affine_relu_cache = affine_relu_forward(X, W1, b1)
        affine2_out, affine2_cache = affine_forward(affine_relu_out, W2, b2)
        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        N = X.shape[0]
        # 求loss
        # exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
        # probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        # loss = (-1)*sum(np.log(probs[range(N),y]))/N
        # loss += 0.5*(np.sum(np.square(W1))+np.sum(np.square(W2)))
        #
        #
        # dscores = probs
        # dscores[range(N),y] -= 1
        # dscores /= N

        # dh,dW2,db2 = affine_backward(dscores,cache3)
        # dh2 = relu_backward(dh,cache2)
        # dx,dW1,db1 = affine_backward(dh2,cache1)
        # grads['W1'] = dW1
        # grads['b1'] = db1
        # grads['W2'] = dW2
        # grads['b2'] = db2

        loss, dscores = softmax_loss(scores, y)

        # loss加正则化
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        dh, dW2, db2 = affine_backward(dscores, affine2_cache)
        grads['W2'] = dW2 + self.reg * self.params['W2']
        grads['b2'] = db2

        dx, dW1, db1 = affine_relu_backward(dh, affine_relu_cache)
        grads['W1'] = dW1 + self.reg * self.params['W1']
        grads['b1'] = db1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    # 没有正则化的loss
    def loss_without_reg(self, X, y=None):
        scores = None
        loss = 0
        grads = {}
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        # 用layers.py
        # hidden_layer, cache1 = affine_forward(X,W1,b1)
        # hidden_layer, cache2 = relu_forward(hidden_layer)
        # scores, cache3 = affine_forward(hidden_layer,W2,b2)
        #
        # # #求loss
        # # exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
        # # probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        # # loss = (-1)*sum(np.log(probs[range(X.shape[0]),y]))/X.shape[0]
        # # #求梯度
        # # dscores = probs
        # # dscores[range(X.shape[0]),y] -= 1
        # # dscores /= X.shape[0]
        #
        # #用softmax_loss方法
        # loss , dscores = softmax_loss(scores,y)
        #
        # dh, dW2, db2 = affine_backward(dscores,cache3)
        # dh2 = relu_backward(dh,cache2)
        # dx, dW1, db1 = affine_backward(dh2,cache1)
        #
        # grads['W1'] = dW1
        # grads['b1'] = db1
        # grads['W2'] = dW2
        # grads['b2'] = db2

        # 用layers_utils.py
        # 求scores
        affine_relu_out, affine_relu_cache = affine_relu_forward(X, W1, b1)
        affine_out, affine_cache = affine_forward(affine_relu_out, W2, b2)
        scores = affine_out

        # 求loss
        loss, dscores = softmax_loss(scores, y)

        # 求梯度
        dh, dW2, db2 = affine_backward(dscores, affine_cache)
        dx, dW1, db1 = affine_relu_backward(dh, affine_relu_cache)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # i的取值为[1,num_layers],第一层为输入层
        # hidden_dim存放每一层隐藏层的尺寸，前一个数值为每层网络的输入层尺寸大小，后一个数值为该层网络输出层的尺寸大小
        # 最后一层为分类输出层
        for i in range(1, self.num_layers + 1):
            if i == 1:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dims[i - 2]
            if i == self.num_layers:
                layer_output_dim = num_classes
            else:
                layer_output_dim = hidden_dims[i - 1]

            # 初始化每层的w和b
            self.params['W' + str(i)] = np.random.normal(0, weight_scale, (layer_input_dim, layer_output_dim))
            self.params['b' + str(i)] = np.zeros(layer_output_dim)

            # 当使用批量标准化时，初始化每层的gamma和beta
            if use_batchnorm and i != self.num_layers:
                self.params['gamma' + str(i)] = np.ones(layer_output_dim)
                self.params['beta' + str(i)] = np.zeros(layer_output_dim)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        current_input = X
        affine_relu_cache = {}
        affine_bn_relu_cache = {}
        dropout_cache = {}

        # 第一层 -- 第num_layers-1层的前向传递计算
        for i in range(1, self.num_layers):
            # 便于取出Wi和bi
            keyW = 'W' + str(i)
            keyb = 'b' + str(i)
            # 不使用批量标准化
            # 全连接层-->ReLu层
            if not self.use_batchnorm:
                current_input, affine_relu_cache[i] = affine_relu_forward(current_input, self.params[keyW],
                                                                          self.params[keyb])

            # #使用批量标准化
            # 全连接层-->批量标准化-->ReLu层
            else:
                keyGamma = 'gamma' + str(i)
                keyBeta = 'beta' + str(i)
                current_input, affine_bn_relu_cache[i] = affine_bn_relu_forward(current_input, self.params[keyW],
                                                                                self.params[keyb],
                                                                                self.params[keyGamma],
                                                                                self.params[keyBeta],
                                                                                self.bn_params[i - 1])
            # 在每一个relu层之后使用dropout层
            # ReLu层-->Dropout层
            if self.use_dropout:
                current_input, dropout_cache[i] = dropout_forward(current_input, self.dropout_param)

        # 最后一层计算全连接层
        keyW = 'W' + str(self.num_layers)
        keyb = 'b' + str(self.num_layers)
        affine_out, affine_cache = affine_forward(current_input, self.params[keyW], self.params[keyb])
        scores = affine_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 用softmax计算loss
        loss, dscores = softmax_loss(scores, y)
        # loss加上最后一层的l2正则化
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W' + str(self.num_layers)]))

        # 最后一层求梯度
        affine_dx, affine_dw, affint_db = affine_backward(dscores, affine_cache)
        grads['W' + str(self.num_layers)] = affine_dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = affint_db

        for i in range(self.num_layers - 1, 0, -1):
            # 在loss中网络层结构：
            # 全连接层-->批量标准化层（可选）-->ReLu层-->Dropout层（可选）
            # 在反向求导时：必须先Dropout层求导，之后再根据是否使用BN层，选择反向求导工具函数
            if self.use_dropout:
                affine_dx = dropout_backward(affine_dx, dropout_cache[i])

            if not self.use_batchnorm:
                # 计算前num_layers-1层每层的梯度
                affine_dx, affine_dw, affint_db = affine_relu_backward(affine_dx, affine_relu_cache[i])

            else:
                affine_dx, affine_dw, affint_db, dgamma, dbeta = affine_bn_relu_backward(affine_dx,
                                                                                         affine_bn_relu_cache[i])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta

            # 将每层W和b的梯度存到grads中
            keyW = 'W' + str(i)
            keyb = 'b' + str(i)
            grads[keyW] = affine_dw + self.reg * self.params[keyW]
            grads[keyb] = affint_db

            # loss加上前num_layers-1层的l2正则化
            loss += 0.5 * self.reg * np.sum(np.square(self.params[keyW]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
