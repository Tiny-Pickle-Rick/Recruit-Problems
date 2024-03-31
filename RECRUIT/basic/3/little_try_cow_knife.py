import numpy as np
import torch
from torch.utils import data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
"""本程序以CPU为设备"""
def generate(w, b, m):#w,b是给定的参数，m要生成的training set的个数
    X = torch.normal(0, 1, (m, len(w)))#生成正态分布的X（在这里是2*num矩阵），均值为0，标准差为1
    y = torch.matmul(X, w) + b#完全符合线性关系的y。可惜现实中碰不到
    y += torch.normal(0, 1, y.shape)#加噪声
    return X,y.reshape((-1, 1))#把y变成m维的列向量
def squared_loss(y_hat, y):#y_hat:计算出的标签列向量,y:真实的标签列向量。
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 #/2是为了求导方便

x,y=generate(torch.tensor([3],dtype=torch.float32),5,1000)
net=LinearRegression()

loss=squared_loss
net.fit(x,y)
w=net.coef_[0][0]
b=net.intercept_[0]

_X=torch.tensor([[-4.0],[4.0]])
_Y=net.predict(_X)

plt.scatter(x,y)
plt.plot(_X,_Y,'r',linewidth=2,label=f"y={w}x+{b}")
plt.title('Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()