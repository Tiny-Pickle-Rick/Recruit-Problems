import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
"""本程序以CPU为设备"""
"""本程序比较完全地展现梯度下降的过程"""

"""iterator，yield小批量的数据（即随机选中的batch_size个features、label对）"""
def data_iter(batch_size, features, labels):#batch_size是批量大小，后面两个是特征、标签。
    m = len(features)#整个training set的个数
    indices = list(range(m))# 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)#indices是打乱后的下标
    for i in range(0, m, batch_size):#这一步有小缺点：不一定能完全利用所有的training set(当m%batch_size!=0)
        batch_indices = torch.tensor(indices[i: min(i + batch_size, m)])#batch_indices是打乱后的小批量的下标
        yield features[batch_indices], labels[batch_indices]#返回的是两个张量：随机选中的batch_size个features、label对


"""线性回归模型，就是预测y的函数。w、b要通过梯度下降法来找"""
def linreg(X, w, b):#X:特征矩阵，w：权重矩阵（列向量），b:偏置
    return torch.matmul(X, w) + b

"""代价函数。最常见的是均方损失。这里就是。参数是张量"""
"""这个函数不求平均值，可以减少一点计算量。不过也不多"""
def squared_loss(y_hat, y):#y_hat:计算出的标签列向量,y:真实的标签列向量。
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 #/2是为了求导方便

"""核心代码1：小批量随机梯度下降（SGD）"""
def sgd(params, lr, batch_size):#params：参数，lr：学习率，batch_size：小批量大小
    with torch.no_grad():#为什么要不纳入计算梯度？因为后面对参数求梯度的，只有代价函数，不能影响！
        for param in params:
            param -= lr * param.grad / batch_size#梯度下降（走一小步）
            param.grad.zero_()#更新梯度为0


"""以下为main"""
m=1000
batch_size = 10#超参数：小批量大小
yita = 0.005#超参数：学习率
n = 20#迭代次数。循环边界有两种可能的条件：1.次数到了上限。2.误差已经足够小。这里选第一种
net = linreg#设定模型为线性回归模型。以后好改
loss = squared_loss#设定代价函数由均方误差构成。以后好改
#人造参数
all=np.genfromtxt('training_set_for_basic_learning.csv', delimiter=',')
features=torch.from_numpy(all[:,0:2]).float()
labels=torch.from_numpy(all[:,2]).float()

init_w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)#从N(0,0.0001)中选两个
init_b = torch.zeros(1, requires_grad=True)#b设置为0


"""第一个关系：loss和迭代次数n"""
w=init_w
b=init_b
#对两个参数都要求导
l_n=[]#存储对应的损失函数值！
"""以下是把没有经过梯度下降损失的计入train_l[0]"""
with torch.no_grad():  # 不纳入梯度计算
    l_n.append((loss(net(features, w, b), labels)).mean())
"""核心代码2：一共进行times轮随机梯度下降，每一轮要用几乎（不排除有剩余的）整个training set算一次梯度，但是分批进行（不然太慢）"""
for i in range(n):#进行n轮梯度下降
    for X, y in data_iter(batch_size, features, labels):#从整个training set中取一个小批量
        l = loss(net(X, w, b), y) #X和y的小批量损失
        l.sum().backward()#先求和，再反向传播！
        sgd([w, b],yita, batch_size) #使用参数的梯度更新参数，也就是：依靠这个小批量算出的梯度下降一步
    with torch.no_grad():#不纳入梯度计算
        l_n.append((loss(net(features, w, b), labels)).mean())#第i次迭代后，对于整个训练集的平均误差
        print(f'epoch={i + 1}, loss={float(l_n[i+1]):f}')

plt.figure(figsize=(8, 4),dpi=100)#设置图片的大小、像素
plt.plot(list(range(0,n+1)),l_n)#显示一次数据点图
plt.xlabel('n')
plt.ylabel(f'Loss when learning rate={yita:f} and batch size={batch_size}')
plt.title('Relations between loss and n')
plt.show()

"""第二个关系：loss和学习率"""
lr=list(np.arange(0.0001,0.0011,0.0001))
l_lr=[]
for yita in lr:
    w=init_w
    b=init_b
    for i in range(n):
        for X,y in data_iter(batch_size,features,labels):
            l=loss(net(X,w,b),y)
            l.sum().backward()
            sgd([w,b],yita,batch_size)
    with torch.no_grad():
        l_lr.append((loss(net(features, w, b), labels)).mean())#这个是对于各个学习率的分别的损失。
        print(f'learning rate={yita:f},loss={float(l_lr[-1]):f}')

plt.figure(figsize=(8,4),dpi=100)
plt.plot(lr,l_lr)
plt.xlabel("Learning rate")
plt.ylabel(f"loss after {n} times when batch size={batch_size}")
plt.title('Relations between loss and learning rate')
plt.show()


"""第三个关系：loss和批量大小"""
yita = 0.0005
n=3
bs=list(range(1,251,1))#+list(range(20,50,2))+list(range(50,255,5))
l_bs=[]
for batch_size in bs:
    w=init_w
    b=init_b
    for i in range(n):
        for X,y in data_iter(batch_size,features,labels):
            l=loss(net(X,w,b),y)
            l.sum().backward()
            sgd([w,b],yita,batch_size)
    with torch.no_grad():
        l_bs.append((loss(net(features,w,b),labels)).mean())
        print(f'batch size={batch_size},loss={float(l_bs[-1]):f}')


plt.figure(figsize=(8,4),dpi=100)
plt.plot(bs,l_bs)
plt.xlabel("batch size")
plt.ylabel(f"loss after {n} times when learning rate={yita:f}")
plt.title('Relations between loss and batch size')
plt.show()
