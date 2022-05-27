
# 融入 局部归一化 
# 融入 Dropout机制 对于全连接层以一定概率丢弃一些值，不参与反向传播计算，减少过拟合；卷积层有池化层来解决过拟合


import torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet,self).__init__()
    # 3,224,224 -> 96,55,55 Input 3 Output 96 
    self.conv1 = nn.Sequential(
        nn.Conv2d(3,96,kernel_size=11,stride=2,padding=2),
        nn.LocalResponseNorm(1),
        nn.ReLU())
    # 96,55,55 -> 128,27,27 Input 96 Output 256
    self.conv2 = nn.Sequential(
        nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),
        nn.LocalResponseNorm(1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2)
    )
    # 128,27,27 -> 384,13,13 Input 256 Output 384
    self.conv3 = nn.Sequential(
        nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
        nn.LocalResponseNorm(1),
        nn.ReLU()
    )
    # 383,13,13 -> 384,13,13 Input 384 Output 384
    self.conv4 = nn.Sequential(
        nn.Conv2d(384,384,kernel_size=5,stride=1,padding=2),
        nn.LocalResponseNorm(1),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=3,stride=2)
    )
    # 384,13,13 -> 256,6,6 Input 384 Output 256
    self.conv5 = nn.Sequential(
        nn.Conv2d(384,256,kernel_size=5,stride=1,padding=2),
        nn.LocalResponseNorm(1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2)
    )
    # Input 6*6*256 Output 4096
    self.dense1 = nn.Sequential(
        nn.Linear(6*6*256,4096),
        nn.ReLU(),
        nn.Dropout(p=0.5)
    )
    # Input 4096 Output 4096
    self.dense2 = nn.Sequential(
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(p=0.5)
    )
    #
    self.dense3 = nn.Linear(4096,1000) # 

    def forward(self,x):
      x = self.conv1(x) # 3,224,224
      x = self.conv2(x) # 96,55,55
      x = self.conv3(x) # 256,27,27
      x = self.conv4(x) # 384,13,13
      x = self.conv5(x) # 384,13,13

      x = x.view(x.size(0),-1)

      x = self.dense1(x) # 256*6*6 -> 4096
      x = self.dense2(x) # 4096 -> 4096
      x = self.dense3(x) # 4096 -> 1000

      return x

alexnet = AlexNet()
print(alexnet)    
