import tensorflow as tf

import numpy as np
import matplotlib as plt

seed = 2

REGULARIZER = 0.01

# 定义神经网络的输入、参数和输出，定义前向传播过程
def get_weights(shape,regularizer):
    w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    # 对参数做正则化处理，防止过拟合
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

def forward(x,regularizer):
    w1 = get_weights([2,11],regularizer)
    b1 = get_bias([11])
    # 中间层的预测输出　使用激活函数relu
    y1 = tf.nn.relu(tf.matmul(x,w1) + b1)

    w2 = get_weights([11,1],regularizer)
    b2 = get_bias([11])
    # 输出层的预测输出
    # 输出层输出不加激活函数
    y = tf.matmul(y1,w2) + b2

    return y

# if __name__=='__main__':
#
#     rdm = np.random.RandomState(seed)
#     X = rdm.randn(100,2)
#     from IPython import embed
#     embed()
#     forward(X,0.01)