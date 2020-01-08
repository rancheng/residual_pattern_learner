import torch
import torch.nn as nn
import torch.functional as F


class PatternNetBase(nn.Module):

    def __init__(self, conf, num_points):
        """Base class for residual pattern module
        args:
            conf: configuration yaml file
        """
        super().__init__()
        
        self.inc = ResBlock(conf['in_channels'], 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        """Default residual pattern index segmented through U-Net 
        with softmax output over the channels.

        args:
            x: image tensor (bs, in_channels, h, w)
        returns:
            seg: lower resolution segmentation (bs, n_classes, h/4, w/4)
        """
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x4)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x = self.outc(x7)
        return torch.sigmoid(x)


# U-Net Parts

class OutConv(nn.Module):
    """Output Convolution"""
    def __init__(self, in_c, n_classes):
        self.conv = nn.Conv2d(in_c, n_classes, 1, 1)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Pool => Conv Block"""
    def __init__(self, in_c, out_c, pool='max', conv='residual'):
        super().__init__()

        # pooling operation
        if pool == 'average':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # convolutional block
        if conv == 'residual':
            self.conv = ResBlock(in_c, out_c)
        else:
            self.conv = DoubleConv(in_c, out_c)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Tranposed Conv Upsample => Conv Block"""
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, in_c, 2, 2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.add(x1, x2)
        return self.conv(x)


class ResBlock(nn.Module):
    """Full pre-activation variant of Residual Block"""

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            
        )
        self.linear = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        return torch.add(self.conv(x), self.linear(x))


class DoubleConv(nn.Module):
    """Standard Double Conv Block: (Conv => BatchNorm => Relu) * 2"""

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1).
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
