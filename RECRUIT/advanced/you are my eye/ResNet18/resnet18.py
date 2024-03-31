import torch
import torchvision
import time
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
"""以下3个函数是我自己写的，在train_show.py中被定义"""
from train_show import get_set,train,show
device="cuda:0"

"""本程序基于预训练的resnet18实现图片分类"""


"""ii.预处理（核心代码在get_set()函数中，它在train_show.py中被定义）"""
train_set, test_set = get_set()
batch_size = 32
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size,drop_last=False)


"""下面为训练做一些准备工作："""
#实例化net
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#以下两行代码把net的全连接层改为输出为10个
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
#训练后参数存放的文件名
save_dir='resnet18.pth'
trained=0#手动调整，这次要不要读入上次参数。当然，第一次训练就trained=0
if trained:
    state_dict = torch.load(save_dir)
    net.load_state_dict(state_dict['net'])

#下面分开特征提取层的参数、前面的参数，用不同的学习率
#得到特征提取层的参数的地址
output_id = list(map(id, net.fc.parameters()))
#分开特征提取层的参数、全连接层的参数
#实际上这种写法完全是多此一举。调用optim.SGD()时第一个实参params用字典构成的列表，完全可以达到目的。见vgg13.py
feature_params = filter(lambda p: id(p) not in output_id, net.parameters())
#一些超参数，可手动调整：
lr, num_epochs, weight_decay=0.01,10,0.001
#优化算法：
optimizer = optim.SGD([{'params': feature_params},
                       {'params': net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=weight_decay)
#训练过程中调整学习率的方案（指数衰退&余弦退火），可以手动调整：
#scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  num_epochs)
#scheduler=None

"""iii.训练模型（核心代码在train()函数中，它在train_show.py中被定义）"""
train_loss ,train_acc,test_acc=train(net, train_iter, test_iter, optimizer, num_epochs,save_path=save_dir,scheduler=scheduler)


"""iv.模型评估&可视化（核心代码在show()函数中，它在train_show.py中被定义）"""
show(num_epochs,train_loss,train_acc,test_acc,'ResNet18')