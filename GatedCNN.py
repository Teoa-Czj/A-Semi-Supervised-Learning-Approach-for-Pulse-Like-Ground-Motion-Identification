from torch import nn
import torch

# 定义 Gated 卷积层
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedConv2d, self).__init__()
        # out_channels 是两倍，因为需要生成候选激活值和门控信号
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # 将输出分为候选激活值和门控信号两部分
        c, g = torch.chunk(x, 2, dim=1)
        g = self.sigmoid(g)
        x = g * c
        return x

def GatedCNNModel(num_classes=1, output='logits'):
    if output == 'logits':
        net = nn.Sequential(GatedConv2d(1,32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            GatedConv2d(32,64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.LazyLinear(128), nn.ReLU(),
                            nn.LazyLinear(64),nn.ReLU(),
                            nn.LazyLinear(num_classes)
                            )
    if output == 'probs':
        net = nn.Sequential(GatedConv2d(1,32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            GatedConv2d(32,64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.LazyLinear(128), nn.ReLU(),
                            nn.LazyLinear(64),nn.ReLU(),
                            nn.LazyLinear(num_classes),
                            nn.Sigmoid()
                            )
    return net

def CNNModel(num_classes=1, output='logits'):
    if output == 'logits':
        net = nn.Sequential(nn.Conv2d(1,32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32,64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.LazyLinear(128), nn.ReLU(),
                            nn.LazyLinear(64),nn.ReLU(),
                            nn.LazyLinear(num_classes)
                            )
    if output == 'probs':
        net = nn.Sequential(nn.Conv2d(1,32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(32,64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Flatten(),
                            nn.LazyLinear(128), nn.ReLU(),
                            nn.LazyLinear(64),nn.ReLU(),
                            nn.LazyLinear(num_classes),
                            nn.Sigmoid()
                            )
    return net