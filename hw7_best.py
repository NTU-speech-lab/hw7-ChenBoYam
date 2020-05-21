# -*- coding: utf-8 -*-
# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import sys
"""#Read image
利用 OpenCV (cv2) 讀入照片並存放在 numpy array 中
"""
workspace_dir = sys.argv[1]
output_dir = sys.argv[2]

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

# 分別將 training set、validation set、testing set 用 readfile 函式讀進來
#workspace_dir = './food-11'
#print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
#print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
#print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
#print("Size of Testing data = {}".format(len(test_x)))

"""# Dataset
在 PyTorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。

Dataset 需要 overload 兩個函數：\_\_len\_\_ 及 \_\_getitem\_\_

\_\_len\_\_ 必須要回傳 dataset 的大小，而 \_\_getitem\_\_ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。

實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。
"""

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

"""# Model"""

class Classifier(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(Classifier, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

"""# Testing
利用剛剛 train 好的 model 進行 prediction
"""
import pickle
def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict
student_net = Classifier(base=16).cuda()
student_net.load_state_dict(decode8('8_bit_model.pkl'))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

student_net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = student_net(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open(output_dir, 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))