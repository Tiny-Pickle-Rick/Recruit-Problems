import torch
import csv

"""用来随机生成training set"""
def generate(w, b, m,std1,std2):#w,b是给定的参数，m要生成的training set的个数
    X = torch.normal(0, std1, (m, len(w)))#生成正态分布的X（在这里是2*num矩阵），均值为0，标准差为1
    y = torch.matmul(X, w) + b#完全符合线性关系的y。可惜现实中碰不到
    y += torch.normal(0, std2, y.shape)#加噪声
    return X,y.reshape((-1, 1))#把y变成m维的列向量

#人造参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
m=1000
features, labels = generate(true_w, true_b, m,1,0.01)
X=torch.cat((features,labels),1).tolist()
with open('training_set_for_basic_learning.csv', 'w', encoding='utf-8', newline='') as f:
    csv_write=csv.writer(f)
    for i in range(m):
        csv_write.writerow(X[i])