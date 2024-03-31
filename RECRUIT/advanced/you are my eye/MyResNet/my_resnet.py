import torch
import torchvision
import time
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
"""以下3个函数是我自己写的，在train_show.py中被定义"""
from train_show import get_set,train,show
import matplotlib.pyplot as plt

device="cuda:0"

"""本程序用自己搭建的resnet18实现图片分类"""


"""下面搭resnet："""
class Residual(nn.Module):#Residual，残差神经网络的基本单元
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,first_block=False): #residual块，严格来说应该叫做“stage”
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block: #意思是：若不是第一个块，都需要通道数翻倍、高宽减半。
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels)) #意思是：第一个块不需要通道数翻倍、高宽减半。
    return blk

#5个stage
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),#这些全是套路了，我没改它
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

#以下是5个stage连起来，加一个全局平均池化层，加一个全连接层
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))


"""ii.数据预处理（核心代码在get_set()函数中，它在train_show.py中被定义）"""
train_set, test_set = get_set(size=32)
batch_size = 32
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size)


"""一些准备工作："""
#训练后参数存放的文件名
save_dir='my_resnet.pth'
trained=0#手动调整，这次要不要读入上次参数。当然，第一次训练就trained=0
if trained:
    state_dict = torch.load(save_dir)
    net.load_state_dict(state_dict['net'])
#一些超参数，可手动调整：
lr,num_epochs = 0.01,10
#优化算法：
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.001)
#训练过程中调整学习率的方案（指数衰退&余弦退火），可手动调整：
#scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,0.8)
#scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  num_epochs)
scheduler=None


"""iii.模型训练（核心代码在train()函数中，它在train_show.py中被定义）"""
train_loss ,train_acc,test_acc=train(net, train_iter, test_iter, optimizer, num_epochs,save_path=save_dir,scheduler=scheduler)


"""iv.模型评估&可视化（核心代码在show()函数中，它在train_show.py中被定义）"""
show(num_epochs,train_loss,train_acc,test_acc,'my resnet')