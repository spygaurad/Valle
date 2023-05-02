import torch
import torch.nn as nn
from torchvision import models

from src.model.handwriting.transformer import TransformerDecoderLayer, TransformerDecoder


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)]
    ])[activation]


def conv_bn(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, 3, 1, 1),
            activation_func(activation),
            conv_bn(self.out_channels, self.out_channels, 3, 1, 1)
        )

        self.activate = activation_func(activation)

    def forward(self, x):

        x = self.blocks(x) + x
        x = self.activate(x)
        return x


class TransposedResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.Sequential(nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_func(activation))

    def forward(self, x):

        x = self.blocks(x)

        return x


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n, activation, kernel_size, stride, padding):
        super().__init__()

        self.blocks = nn.Sequential(
            TransposedResNetBasicBlock(in_channels, out_channels, activation, kernel_size, stride, padding),
            *[ResNetBasicBlock(out_channels, out_channels, activation) for _ in range(n-1)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet(nn.Module):
    def __init__(self, channels, depths, activation, kernels_size, strides, paddings):
        super().__init__()

        self.in_out_channel_sizes = list(zip(channels, channels[1:]))

        self.blocks = nn.ModuleList([
            *[ResNetLayer(in_c, out_c, n, activation, kernel_size, stride, padding)
              for (in_c, out_c), n, kernel_size, stride, padding in zip(self.in_out_channel_sizes, depths, kernels_size, strides, paddings)]
        ])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x


# class ImageDecoder(nn.Module):
#     def __init__(self, config):
#         super(ImageDecoder, self).__init__()
# 
#         self.char_feature_map = nn.Sequential(nn.Conv2d(config.char_embedding_dim,
#                                                         config.resnet_decoder['channels'][0],
#                                                         kernel_size=1,
#                                                         bias=True),
#                                               activation_func(config.resnet_decoder['activation']))
# 
#         self.initial_channels = config.resnet_decoder['channels'][0]
# 
#         self.reshape_h = config.resnet_decoder['reshape_size'][0]
#         self.reshape_w = config.resnet_decoder['reshape_size'][1]
# 
#         self.initial_transposed_conv = nn.Sequential(nn.ConvTranspose2d(
#             config.resnet_decoder['channels'][0],
#             config.resnet_decoder['channels'][0],
#             kernel_size=(2,2),
#             stride=(2,2),
#             padding=(0,0),
#             bias=True),
#             activation_func(config.resnet_decoder['activation']))
# 
#         self.resnet = ResNet(config.resnet_decoder['channels'],
#                              config.resnet_decoder['depths'],
#                              config.resnet_decoder['activation'],
#                              config.resnet_decoder['kernels_size'],
#                              config.resnet_decoder['strides'],
#                              config.resnet_decoder['paddings'])
# 
#         self.final_conv = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, bias=True),
#                                         activation_func(config.resnet_decoder['activation']))
# 
#     def forward(self, char_embedding):
# 
#         char_embedding = char_embedding.permute(0, 2, 1).unsqueeze(dim=-2)
#         char_embedding = self.char_feature_map(char_embedding)
#         char_embedding = char_embedding.view(-1, self.initial_channels, self.reshape_h, self.reshape_w)
# 
#         char_embedding = self.initial_transposed_conv(char_embedding)
# 
#         output = char_embedding
#         output = self.resnet(output)
#         output = self.final_conv(output)
# 
#         return output

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, output_size, input_size=2048):
        super(ConditionalBatchNorm2d, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

        self.gain = nn.Linear(self.input_size, self.output_size)
        self.bias = nn.Linear(self.input_size, self.output_size)

        # self.gain = nn.Sequential(
        #                     nn.Linear(self.input_size, self.output_size),
        #                     nn.ReLU(inplace=True),
        #                     nn.Linear(self.output_size, self.output_size))

        # self.bias = nn.Sequential(
        #                     nn.Linear(self.input_size, self.output_size),
        #                     nn.ReLU(inplace=True),
        #                     nn.Linear(self.output_size, self.output_size))


        self.batch_norm2d = nn.BatchNorm2d(output_size, affine=False)



    def forward(self, x, y):
        gain = (self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        out = self.batch_norm2d(x)

        return out * gain + bias

class UpSampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=(1,2)):
        super(UpSampleBlock, self).__init__()
        
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=upsample)
        
        # self.batchnorm_in = nn.BatchNorm2d(in_channel)
        # self.batchnorm_out = nn.BatchNorm2d(out_channel)
        self.batchnorm_in = ConditionalBatchNorm2d(in_channel)
        self.batchnorm_out = ConditionalBatchNorm2d(out_channel)

        self.conv1by1 = nn.Conv2d(in_channel, out_channel, 1)
        self.conv3by3_1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv3by3_2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, feature_map, conditional_embedding):
        feature_map = self.batchnorm_in(feature_map, conditional_embedding)
        feature_map = self.relu(feature_map)
        feature_map = self.upsample(feature_map)

        feature_map_3by3 = self.conv3by3_1(feature_map)
        feature_map_3by3 = self.batchnorm_out(feature_map_3by3, conditional_embedding)
        feature_map_3by3 = self.relu(feature_map_3by3)
        feature_map_3by3 = self.conv3by3_2(feature_map_3by3)

        feature_map_1by1 = self.conv1by1(feature_map)

        feature_map = feature_map_3by3 + feature_map_1by1

        return feature_map

class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.upsample_128_64 = UpSampleBlock(128, 64) 
        self.upsample_64_32 = UpSampleBlock(64, 32) 
        self.upsample_32_8 = UpSampleBlock(32, 8) 
        self.upsample_8_1 = UpSampleBlock(8, 1) 
        self.conv3by3 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, feature_map, conditional_embedding):
        feature_map = self.upsample_128_64(feature_map, conditional_embedding)
        feature_map = self.upsample_64_32(feature_map, conditional_embedding)
        feature_map = self.upsample_32_8(feature_map, conditional_embedding)
        feature_map = self.upsample_8_1(feature_map, conditional_embedding)
        feature_map = self.conv3by3(feature_map)

        return feature_map


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

class FCN_Decoder_Style(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=256, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(FCN_Decoder_Style, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        # for i in range(ups):
        #     self.model += [nn.Upsample(scale_factor=2),
        #                    Conv2dBlock(dim, dim // 2, 5, 1, 2,
        #                                norm='in',
        #                                activation=activ,
        #                                pad_type=pad_type)]
        #     dim //= 2
        # self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
        #                            norm='none',
        #                            activation='tanh',
        #                            pad_type=pad_type)]

        channels = [256, 128, 64, 32, 16, 8, 1]
        # input: 4 * 256
        # output: 8 * 512
        self.model += [nn.Upsample(scale_factor=(2,2)),
                       Conv2dBlock(channels[0], channels[1], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        # input: 8 * 512
        # output: 16 * 512
        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[1], channels[2], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        # input: 16 * 512
        # output: 32 * 1024
        self.model += [nn.Upsample(scale_factor=(2,2)),
                       Conv2dBlock(channels[2], channels[3], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        # input: 32 * 1024
        # output: 64 * 1024
        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[3], channels[4], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        # input: 64 * 1024
        # output: 128 * 2048
        self.model += [nn.Upsample(scale_factor=(2,2)),
                       Conv2dBlock(channels[4], channels[5], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        #No change
        self.model += [Conv2dBlock(channels[5], 1, 7, 1, 3,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
        #No change
        self.model += [nn.Conv2d(1, 1, 3, padding=1)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y =  self.model(x)

        return y


class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=128, out_dim=1, res_norm='in', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        # for i in range(ups):
        #     self.model += [nn.Upsample(scale_factor=2),
        #                    Conv2dBlock(dim, dim // 2, 5, 1, 2,
        #                                norm='in',
        #                                activation=activ,
        #                                pad_type=pad_type)]
        #     dim //= 2

        # channels = [512, 64, 32, 8, 4, 2, 1]
        # channels = [128, 128, 64, 32, 16, 8, 1]
        """
        channels = [512, 256, 128, 64, 32, 16, 1]
        self.model += [nn.Upsample(scale_factor=(4,1)),
                       Conv2dBlock(channels[0], channels[1], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(1,2)),
                       Conv2dBlock(channels[1], channels[2], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
                       
        self.model += [nn.Upsample(scale_factor=(4,1)),
                       Conv2dBlock(channels[2], channels[3], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(1,2)),
                       Conv2dBlock(channels[3], channels[4], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2, 1)),
                       Conv2dBlock(channels[4], channels[5], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [Conv2dBlock(channels[5], 1, 7, 1, 3,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]


        self.model += [nn.Conv2d(1, 1, 3, padding=1)]
        """


        """
        #Generate 32*512 image
        channels = [512, 128, 64, 32, 16, 1]
        # channels = [256, 128, 64, 32, 16, 1]
        self.model += [nn.Upsample(scale_factor=(1,2)),
                       Conv2dBlock(channels[0], channels[1], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[1], channels[2], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]
                       
        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[2], channels[3], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[3], channels[4], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [Conv2dBlock(channels[4], 1, 7, 1, 3,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Conv2d(1, 1, 3, padding=1)]

        """
        #Generate 32*1024 image
        channels = [512, 128, 64, 32, 16, 8, 1]
        self.model += [nn.Upsample(scale_factor=(1,2)),
                       Conv2dBlock(channels[0], channels[1], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[1], channels[2], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(1,2)),
                       Conv2dBlock(channels[2], channels[3], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[3], channels[4], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Upsample(scale_factor=(2,1)),
                       Conv2dBlock(channels[4], channels[5], 5, 1, 2,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [Conv2dBlock(channels[5], 1, 7, 1, 3,
                                   norm='in',
                                   activation=activ,
                                   pad_type=pad_type)]

        self.model += [nn.Conv2d(1, 1, 3, padding=1)]
        # """


        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        y =  self.model(x)
        return y


class ImageDecoder(nn.Module):

    def __init__(self, vocab_size, char_embedding, pos_encoding, config, device):
        super(ImageDecoder, self).__init__()

        self.device = device
        # self.char_embedding = char_embedding
        self.pos_encoding = pos_encoding

        self.dropout = nn.Dropout(p=config.transformer_decoder['dropout'])

        encoder_layer = nn.TransformerEncoderLayer(d_model=512,
                                                   nhead=config.transformer_encoder['num_heads'],
                                                   dropout=config.transformer_decoder['dropout'],
                                                   dim_feedforward=config.transformer_decoder['ffn'],
                                                   norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=1)
        # """ 
        decoder_layer = nn.TransformerDecoderLayer(d_model=512,
                                                   nhead=config.transformer_decoder['num_heads'],
                                                   dropout=config.transformer_decoder['dropout'],
                                                   dim_feedforward=config.transformer_decoder['ffn'],
                                                   norm_first=False)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=1)
        """
        decoder_layer = TransformerDecoderLayer(512, 8, 1024, 0.5, "relu", True)
        decoder_norm = nn.LayerNorm(512)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 1, decoder_norm, return_intermediate=False)
       #  """

        self.style_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(pretrained=True).children())[1:-2]))

        # self.style_linear = nn.Linear(512, 2048)

        self.linear = nn.Linear(2048, 512)
        # self.linear_2 = nn.Linear(512, 8192)
        self.linear_2 = nn.Linear(512, 4096) 
        # self.linear_2 = nn.Linear(512, 2048) 
        self.dropout_linear = nn.Dropout(p=0.5)
        # self.linear_2 = nn.Linear(1024, 4096)
        # self.linear_2 = nn.Linear(512, 2048)
        self.activation_function = nn.ReLU()
        self.tanh = nn.Tanh()


        self.conv1d = nn.Conv1d(1, 128, 385)

        self.upsample = UpSample()

        self.batchnorm = nn.BatchNorm2d(128)

        self.initial_channel = 512

        self.DEC = FCNDecoder(dim=self.initial_channel)



    def forward(self, tgt=None, style=None, tgt_key_padding_mask=None, return_style_only=False):


        if style is not None:
            style = self.style_encoder(style)
            style = style.flatten(2).permute(0, 2, 1)
            # style = self.style_linear(style)
            style = style.permute(1, 0, 2)
            style = self.transformer_encoder(style)

            if return_style_only:
                style = style.permute(1, 0, 2)
                return style

            tgt = tgt.permute(1, 0, 2)

            output = self.transformer_decoder(
                tgt=tgt,
                memory=style
            )

        else:
            print("There must always be a style")

        # output = output.squeeze(0)

        output = output.permute(1, 0, 2)

        # print("output:", output.shape)
        # print("tgt: ", tgt.shape)
        # print("cat:", torch.cat([output, tgt.permute(1,0,2)],-1).shape)
        # print("\n")


        # output = torch.cat([output, tgt.permute(1,0,2)],-1)

        # output = self.dropout_linear(self.linear_2(output))
        output = (self.linear_2(output))
        output = output.contiguous()

        bs, seq_len, embed_dim = output.shape

        output = output.view(bs, self.initial_channel, 4, -1)
        # print(output.shape)

        output = self.DEC(output)
        # print(output.shape)
        # output = self.tanh(output)
        if style is not None: 
            style = style.permute(1,0,2)
            return style, output
        else:
            print("There must always be a style")

    def generate(self, tgt, tgt_key_padding_mask=None):

        # tgt = self.linear(tgt)

        # tgt_mask = tgt_mask[0]

        # tgt = tgt  + self.pos_encoding(tgt) #self.dropout(self.char_embedding(tgt) + self.pos_encoding(tgt))

        tgt = tgt.reshape(-1, 128, 16, 128)
        tgt = tgt.reshape(-1, 2048, 128)
        tgt = tgt + self.pos_encoding(tgt)

        conditional_embedding = torch.mean(tgt, dim=1)

        tgt = tgt.permute(1, 0, 2)


        output = self.transformer_decoder(
            src=tgt,
            src_key_padding_mask=tgt_key_padding_mask #To avoid looking at pad token
        )

        output = output.permute(1, 0, 2)

        output = self.activation_function(output)

        output = output.reshape(-1, 128, 2048)

        output = output.unsqueeze(1)
        # output = torch.sum(output, dim=1)
        # output = torch.reshape(output, (-1, 16, 256)).unsqueeze(1)

        # output = self.linear(output)
        # print(output.shape)

        return output


