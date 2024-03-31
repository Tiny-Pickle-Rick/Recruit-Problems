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

"""本程序基于预训练的VGG13实现图片分类"""


"""ii.数据预处理（核心代码在get_set()函数中，它在train_show.py中被定义）"""
train_set, test_set = get_set(size=32)
batch_size = 32
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size,drop_last=False)


"""下面实例化VGG13_bn类，并改动它"""
#实例化类
net=models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
#冻结特征提取层。可手动取消。若取消，则要取消第41行的注释
#for param in net.features.parameters():
#    param.requires_grad_(False)
#把最后一个全连接层输出改为10个
net.classifier[6]=nn.Linear(4096,10,True)


"""训练前的一些准备工作"""
#训练后参数存放的文件名
save_dir='vgg13.pth'
trained=0#手动调整，这次要不要读入上次参数。当然，第一次训练就trained=0
if trained:
    state_dict = torch.load(save_dir)
    net.load_state_dict(state_dict['net'])
#一些超参数，可手动调整：
num_epochs=10
lr=0.01
#下一行的{'params': net.features.parameters()},可取消注释，恢复，同时取消第25行的“冻结特征提取”
optimizer=optim.SGD([{'params': net.features.parameters()},
                       {'params': net.classifier.parameters(), 'lr': lr * 10}],lr=lr,weight_decay=1e-3)
#训练过程中调整学习率的方案（指数衰退&余弦退火），可手动调整：
#scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  num_epochs)
#scheduler=None


"""iii.模型训练（核心代码在train()函数中，它在train_show.py中被定义）"""
train_loss ,train_acc,test_acc=train(net=net, train_iter=train_iter, test_iter=test_iter, optimizer=optimizer, num_epochs=num_epochs,save_path=save_dir,scheduler=scheduler)


"""iv.模型评估&可视化（核心代码在show()函数中，它在train_show.py中被定义）"""
show(num_epochs,train_loss,train_acc,test_acc,'VGG13')