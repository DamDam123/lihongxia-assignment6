# coding=utf-8
from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]

    # 把x转为二维向量（N,D = d_1 * ... * d_k）赋值给新变量x_new，否则在backward中x的形状变化
    x_new = x.reshape(N, -1)
    # out = x*w + b
    out = x_new.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)上游梯度
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N = x.shape[0]
    x_new = np.reshape(x, (N, -1))
    # 本地梯度等于上游梯度乘以本地求导
    # x的本地求导是w
    dx = dout.dot(w.T)
    # 将dx转为和x形状相同
    dx = np.reshape(dx, x.shape)
    # w的本地求导是x
    dw = x_new.T.dot(dout)
    # db本地求导为（M,）形状的单位矩阵，乘以上游梯度dout后，即为上游梯度每一列的求和
    db = np.sum(dout, axis=0, keepdims=True)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # ReLU: f(x) = max(0,x)
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # dx在本地导数为1，因此乘以上游梯度等于上游梯度
    dx = dout
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


# 白化处理：对输入数据分布变为均值为0，方差为1的正态分布--神经网络会较快收敛
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        # 将数据的分布归一化为均值为0方差为1的分布
        # x^ = (x-E(x)/√Var(x))

        x_sample_mean = np.mean(x, axis=0)  # 均值
        x_sample_var = np.var(x, axis=0)  # 方差
        # 形成均值为0，方差为1的正态分布,x = (x - mean(x))/sqrt(方差（x) + eps)
        x_normalized = (x - x_sample_mean) / np.sqrt(x_sample_var + eps)
        # y = gamma * x_normalized + beta
        out = gamma * x_normalized + beta

        running_mean = momentum * running_mean + (1 - momentum) * x_sample_mean
        running_var = momentum * running_var + (1 - momentum) * x_sample_var

        cache = (x, x_sample_mean, x_sample_var, x_normalized, gamma, beta, eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # 归一化处理x^ = (x-E(x)/√Var(x))
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

        out = gamma * x_normalized + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


# 画计算图，用反向传播一步一步向前推
# 参考：https://blog.csdn.net/xiaojiajia007/article/details/54924959
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    (x, x_sample_mean, x_sample_var, x_normalized, gamma, beta, eps) = cache

    # (*)乘法运算：对数组执行对应位置相乘；对矩阵执行矩阵乘法运算
    # np.dot()对秩为1的数组，执行对应位置相乘，再相加；对秩不为1的二维数组，执行矩阵乘法运算
    N, D = x.shape
    # beta求导为1
    dbeta = np.sum(dout, axis=0)

    # gamma求导为x_normalized,再乘以上游梯度
    dgamma = np.sum(dout * x_normalized, axis=0)
    # x_normalized求导为gamma，再乘以上游梯度
    dx_normalized = gamma * dout

    # dxivar = np.sum(dx_normalized*(x-x_sample_mean),axis=0)#对于分母
    dxmu1 = dx_normalized / np.sqrt(x_sample_var + eps)#对于分子

    # dsqrtvar = -1./np.sqrt(x_sample_var + eps)**2*dxivar
    #
    # dvar = 0.5*1/np.sqrt(x_sample_var+eps)*dsqrtvar

    dvar = np.sum(-1.0 / 2 * (dx_normalized * (x - x_sample_mean)) * (x_sample_var + eps) ** (-3.0 / 2), axis=0)

    dsp = 1 / N * np.ones((N, D)) * dvar

    dxmu2 = 2 * (x - x_sample_mean) * dsp

    dx1 = dxmu1 + dxmu2
    dmu = -1 * np.sum(dxmu2 + dxmu1, axis=0)

    dx2 = 1 / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


# 简化方法
def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    (x, x_sample_mean, x_sample_var, x_normalized, gamma, beta, eps) = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)

    dgamma = np.sum(dout * x_normalized, axis=0)

    dx_normalized = gamma * dout
    # dvar中(x_sample_var+eps)为什么没有（-3/2）次方（x_normalized中分母为1/2次方简化了）
    dvar = np.sum(-1.0 / 2 * (dx_normalized * x_normalized) / (x_sample_var + eps), axis=0)
    dmean = np.sum(-1 / np.sqrt(x_sample_var + eps) * dx_normalized, axis=0)
    dmean += dvar * np.sum(-2*(x - x_sample_mean), axis=0)/N
    dx = 1 / np.sqrt(x_sample_var + eps) * dx_normalized + 2.0 / N * (x - x_sample_mean) * dvar + 1.0 / N * dmean
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # dropout以大于概率p舍弃神经元并让其他神经元以概率q=1-p保留
        N, D = x.shape
        # *x.shape相当于like的意思，形如x
        # mask = (np.random.rand(*x.shape) < (1-p))
        # 以概率p舍弃
        mask = (np.random.rand(N, D) < p)
        # print(mask)
        out = x * mask / p
        # print(out)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # 如果为test，则返回输入，且mask为none
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # 根据out = x * mask / p
        # 可得dx = mask/p*上游导数dout
        p = dropout_param['p']
        dx = dout * mask / p
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.#步伐大小
      - 'pad': The number of pixels that will be used to zero-pad the input.#边界填充0

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # np.pad(数组名,((第一行上面填充的行数,最后一行下面填充的行数),(第一列左面填充的列数，最后一列右面填充的列数)),
    #        'constant（表示填充连续一样的值）',constant_values=(前面用x填充,后面用y填充))

    # N表示输入数据的个数，C表示每个数据点的通道（对于图片来说是3，即rgb三通道），H表示高度，W表示宽度
    # F表示卷积核个数，C表示卷积核通道，HH表示卷积核高度，WW表示卷积核宽度
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    # 卷积后新矩阵的大小
    new_H = int((H + 2 * pad - HH) / stride + 1)
    new_W = int((W + 2 * pad - WW) / stride + 1)

    out = np.zeros([N, F, new_H, new_W])

    # 卷积过程参见：https://blog.csdn.net/dajiabudongdao/article/details/77263608
    # 第n个图像
    for n in range(N):
        # 第f个卷积核
        for f in range(F):
            # 初始化每次卷积后的结果
            conv_newH_newW = np.zeros([new_H, new_W])
            # 第c个通道
            for c in range(C):
                # 填充原始矩阵中第c通道层，填充大小为pad,填充值为0
                # 在原始矩阵的上下左右各添加一行
                # 填充是防止卷积后维度降低
                x_pedded = np.pad(x[n, c], ((pad, pad), (pad, pad)), 'constant', constant_values=0)
                # 求解out中第i行第j列元素
                for i in range(new_H):
                    for j in range(new_W):
                        # 在第c通道的填充矩阵x_pedded与第f个卷积核的第c个通道的点乘
                        # c个通道的点乘值累加
                        conv_newH_newW[i, j] += np.sum(
                            x_pedded[i * stride:i * stride + HH, j * stride:j * stride + WW] * w[f, c, :, :])
            # c个通道的卷积值算完后加上bias
            out[n, f] = conv_newH_newW + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, new_H, new_W = dout.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    # 多维x填充
    # pad_width((before1, after1),…(beforeN, afterN))((before1, after1),…(beforeN, afterN))，(beforeN, afterN)(beforeN, afterN)
    # 表示第n维之前和之后填充的宽度。
    # n和c不填充，只填充H和W
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                      'constant',
                      constant_values=0)

    dx_padded = np.zeros_like(x_padded)  # 后期去填充即可得到dx
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):# 第n个图像
        for f in range(F):# 第f个卷积核
            #  print(dout[n,f,:,:])
            # db[f] = np.sum(dout[n,f,:,:])
            for i in range(new_H):
                for j in range(new_W):
                    # out = x * w + b
                    db[f] += dout[n, f, i, j]
                    dw[f] += dout[n, f, i, j] * x_padded[n, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
                    dx_padded[n, :, i * stride: HH + i * stride, j * stride: WW + j * stride] += w[f] * dout[n, f, i, j]

    #去填充
    dx = dx_padded[:, :, pad:pad + H, pad:pad + W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape

    new_H = int((H - pool_height) / stride + 1)
    new_W = int((W - pool_width) / stride + 1)

    out = np.zeros([N, C, new_H, new_W])

    for n in range(N):
        for f in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    # 选择窗口内的最大值，降维
                    out[n, f, i, j] = np.max(x[n, f, i * stride: i * stride + pool_height, j * stride: j * stride + pool_width])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.shape

    new_H = int((H - pool_height) / stride + 1)
    new_W = int((W - pool_width) / stride + 1)

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    x_inner = x[n, c, i * stride: i * stride + pool_height, j * stride: j * stride + pool_width]
                    x_max = np.max(x_inner)
                    dx[n, c, i * stride: i * stride + pool_height, j * stride: j * stride + pool_width] = (x_inner == x_max) * dout[n, c, i, j]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    N, C, H, W = x.shape
    # 将x转为(N*H*W,C)
    x_reshape = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
