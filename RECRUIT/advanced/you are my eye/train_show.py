import time
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets,transforms
'''本程序写通用函数，如读入图片、训练'''
device="cuda:0"
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

"""预处理函数"""
def get_set(size=224):
    train_dir = "../data/train"
    test_dir = "../data/test"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if(size==224):#针对
        train_augs = transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_augs = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_augs = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_augs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    train_set = datasets.ImageFolder(train_dir, transform=train_augs)
    test_set = datasets.ImageFolder(test_dir, transform=test_augs)
    return train_set,test_set

def train(net, train_iter, test_iter, optimizer, num_epochs,save_path,scheduler=None):

    train_loss = []
    train_acc = []
    test_acc = []
    net = net.to(device)

    with torch.no_grad():
        net.eval()  # 评估模式
        test_acc_sum, n2 = 0.0, 0
        for X, y in test_iter:
            test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n2 += y.shape[0]

    acc_max=test_acc_sum/n2
    test_acc.append(acc_max)
    loss=nn.CrossEntropyLoss()
    start = time.time()

    net.train()  # 训练模式
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0 # 记录每一次的loss、training acc,test acc

        for X, y in train_iter:
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        with torch.no_grad():
            net.eval()  # 评估模式
            test_acc_sum, n2=0.0, 0
            for X, y in test_iter:
                test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                n2 += y.shape[0]

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        if scheduler:
            scheduler.step()
        #以下为打印、存loss和acc
        train_loss.append(train_loss_sum / batch_count)
        train_acc.append(train_acc_sum / n)
        if ( test_acc_sum/n2 > acc_max):
            torch.save({'net': net.state_dict()}, save_path)
            acc_max=test_acc_sum/n2
        test_acc.append(test_acc_sum/n2)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec, max test acc %.3f'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc_sum / n2, time.time() - start,
                 acc_max))
    return train_loss,train_acc,test_acc

def show(num_epochs,train_loss,train_acc,test_acc,name):
    n = list(range(1, num_epochs + 1))

    fig,ax=plt.subplots()
    ax.plot(n, train_loss, label='Training Loss', linestyle='-', color='b')
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel("Loss")
    ax.legend()

    ax2=ax.twinx()
    ax2.plot(n, train_acc, label='Training accuracy', linestyle='--', color='m')
    ax2.plot(list(range(0,num_epochs+1)), test_acc, label='Test accuracy', linestyle='-.', color='g')
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.title('Visualized Performance of '+name)
    plt.show()