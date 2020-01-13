import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLearner(nn.Module):

    def __init__(self, in_channels, default_pattern, k, use_attention=False):
        """Class for learning residual pattern weights via joint segmentation
        and gaussian blur attention mechanism
        
        args:
            in_channels: integer, number of channels in the input image (e.g. RGB + Mask = 4)
            default_pattern: default residual pattern with shape (m, m)
            k: after applying weights to default_pattern, take top k scores of scaled pattern
        """
        
        # pattern utility functions helper
        self.pattern = PatternHelper(default_pattern=default_pattern, max_points=k)
        
        # residual U-Net segmentor
        self.segmentor = UNet(in_channels=in_channels)
        
        # attention mechanism (optional)
        self.use_attention = use_attention
        if self.use_attention:
            # m must be an odd number (e.g 3x3, 5x5, 11x11, ...)
            m = default_pattern.shape[0]
            self.attention = nn.Conv2d(1, 1, kernel_size=m, stride=1, bias=False, padding=m//2)


    def forward(self, x, points):
        """Computes updated residual pattern weights for each point of interest
        args:
            x: image tensor with shape (B, C, H, W), B=1 
            points: point coordinate tensor with shape (N, 2) 

        returns:
            weights: residual pattern tensor for each point of interested
                     with shape (N, m, m)
        """
        x = self.segmentor(x)

        if self.use_attention:
            x = self.attention(x)

        weights = self.pattern.forward(x, points)
        return weights

class PatternHelper:

    def __init__(self, default_pattern, max_points, threshold=False):
        """Provides utility functions for computing residual patterns.
        args:
            pattern: default tensor pattern with shape (m, m)
            max_points: number of top scoring elements to select 
                        from the weighted residual pattern
        """
        self.default_pattern = default_pattern
        self.m = default_pattern.shape[0]
        self.max_points = max_points
        self.threshold = threshold

        if self.max_points > self.m * self.m:
            print("Cannot select more points than number of elements in pattern tensor")
            exit(1)

    def compute_pattern(self, weights):
        """Computes the updated residual pattern given weights.
        args:
            weights: np.array with shape (N, m, m)
        """
        N = weights.shape[0]

        # element wise product for pattern scaling
        pattern = self.default_pattern * weights

        # just return thresholded values
        if self.threshold:
            return pattern.round()
        
        # find top k scoring elements
        flattened_weights = pattern.reshape(N, self.m*self.m)
        kth_elem = torch.kthvalue(flattened_weights, self.m * self.m - self.max_points)
        for i in range(N):
            weights[i] = torch.gt(weights[i], kth_elem[i])
        
        return weights.float()

    def img_to_weights(self, weight_map, points):
        """Create (m, m) crops of feature map centered at N points
        args:
            weight_map: tensor of shape (H, W)
            points: tensor of shape (N, 2)
        returns:
            weights: tensor of shape (N, m, m)
        """
        
        N = points.shape[0]
        pad = self.m//2

        # pad weight map in case points is at edge of tensor
        weight_map = F.pad(weight_map, (pad, pad, pad, pad))
        weights = np.zeros((N, self.m, self.m))

        for i in range(N):
            p = points[i]
            weights[i] = self.crop_from_point(weight_map, p, pad)

        return weights

    @staticmethod
    def crop_from_point(weight_map, coords, pad):
        x, y = coords.long()
        weights = weight_map[y-pad:y+pad+1, x-pad:x+pad+1]
        return weights

    def forward(self, img, points):
        weights = self.img_to_weights(img, points)
        return self.compute_pattern(weights)


class UNet(nn.Module):

    def __init__(self, in_channels):
        """Backbone for residual pattern module.
        args:
            in_channels: number of input channels for image tensor
        """
        super().__init__()
        
        self.inc = ResBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        """ Given an image tensor, returns a 1-dimensional feature
        map [0, 1] with the same spatial resolution as the input"""
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
        super().__init__()

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
