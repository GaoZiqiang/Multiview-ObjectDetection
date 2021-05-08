'''
生成数据集模块

'''
import numpy as np
import matplotlib as plt

seed = 2
def generate_datasets():
    # 基于seed生成随机数
    rdm = np.random.RandomState(seed)

    X= rdm.randn(100,2)
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0,x1) in X]
    # 遍历Y_中的每个元素，1:red，其余:blue
    Y_c = [['red' if y else 'blue'] for y in Y_]

    # 对于数据集X和标签Y_进行形状整理  这一个不是太懂
    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)

    # 函数最终返回X Y_ Y_c
    return X,Y_,Y_c
