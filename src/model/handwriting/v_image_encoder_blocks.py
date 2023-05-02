from torch.nn import Module, ModuleList
from torch.nn import Conv2d, InstanceNorm2d, Dropout, Dropout2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, Linear, Sigmoid, ReLU
from torch.nn import ReLU
from torch.nn.functional import pad
import random

class SEBlock(Module):
    def __init__(self, in_channels, reduction) -> None:
        super(SEBlock, self).__init__()
        self.gap = AdaptiveAvgPool2d(1)
        self.linear1 = Linear(in_channels, in_channels//reduction, bias=False)
        self.relu = ReLU()
        self.linear2 = Linear(in_channels//reduction, in_channels, bias=False)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x1 = self.gap(x).squeeze(2).squeeze(2)        
        x1 = self.relu(self.linear1((x1)))
        x1 = self.sigmoid(self.linear2((x1)))[:, :, None, None]
        
        output = x1 * x

        return output

class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class FCN_Encoder(Module):
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])

        # self.final_conv = Conv2d(256, 256, kernel_size=(2,1), stride=1, padding=0, bias=True)
        self.ada_pool = AdaptiveMaxPool2d((1, None))

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt

        x = self.ada_pool(x)
        # x = self.final_conv(x)
        return x

class FCN_Encoder_Style(Module):
    def __init__(self, params):
        super(FCN_Encoder_Style, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        init_blocks_out = x

        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        dscb_blocks_out = x 

        return init_blocks_out, dscb_blocks_out

class FCN_Encoder_Style2(Module):
    def __init__(self, params):
        super(FCN_Encoder_Style2, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 2), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        init_blocks_out = x

        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        dscb_blocks_out = x 

        return init_blocks_out, dscb_blocks_out

class FCN_Encoder_SE_Style(Module):
    def __init__(self, params):
        super(FCN_Encoder_SE_Style, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            SEBlock(16, 16),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            SEBlock(32, 16),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            SEBlock(64, 16),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])
        self.se_blocks = ModuleList([
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(256, 16),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        init_blocks_out = x

        for b, se in zip(self.blocks, self.se_blocks):
            xt = se(b(x))
            x = x + xt if x.size() == xt.size() else xt
        dscb_blocks_out = x 

        return init_blocks_out, dscb_blocks_out

class FCN_Encoder_SE(Module):
    def __init__(self, params):
        super(FCN_Encoder_SE, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            SEBlock(16, 16),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            SEBlock(32, 16),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            SEBlock(64, 16),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            SEBlock(128, 16),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])
        self.se_blocks = ModuleList([
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(128, 16),
            SEBlock(256, 16),
        ])

        self.ada_pool = AdaptiveMaxPool2d((1, None))


    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b, se in zip(self.blocks, self.se_blocks):
            xt = se(b(x))
            x = x + xt if x.size() == xt.size() else xt

        x = self.ada_pool(x)

        return x

class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, pool=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=pool)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x
