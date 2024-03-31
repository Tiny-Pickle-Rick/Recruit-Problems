import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
"""本程序以CPU为设备"""
"""用最小二乘法实现的线性回归"""
class LRbyLSM: # 类LRbyLSM：用最小二乘法实现线性回归的模型
    def train(self,X,y): # 参数类型：numpy array
        self.w=(X.T* X).I * X.T * y # 最小二乘法中核心步骤
    def pre(self,X):

        result=X* self.w
        return np.asarray(result).ravel() # 输出拉成一维数组


data=pd.read_csv("boston.csv") # 读入文件
t=data.sample(len(data),random_state=0)
new_columns=t.columns.insert(0,"add")
# 为什么要在最前面加一列？因为我们的最小二乘法是带偏置的。加一列全为1的，就把算偏置转化为算权重，运算更方便
t=t.reindex(columns=new_columns,fill_value=1)

# 训练集：430个
train_X=t.iloc[:430,:-1]
train_y=t.iloc[:430,-1]
# 测试集：76个
test_X=t.iloc[430:,:-1]
test_y=t.iloc[430:,-1]

net=LRbyLSM() # 实例化类
net.train(np.asmatrix(train_X),np.asmatrix(train_y).reshape(-1,1)) # 先由dataframe转换为array
out=net.pre(np.asmatrix(test_X))


plt.figure(figsize=(16,12))
plt.plot(out,'r--',label='The predicted values')
plt.plot(test_y.values,'b-',label='The true values')
plt.xlabel('order of sample')
plt.title('The fitting result of linear regression by LSM')
plt.legend()
plt.show()

net=nn.Sequential(nn.Linear(10,20),nn.Linear(20,5))
print(net[0].bias.data)
print(net[0].bias.data[1])