import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as f
from src.model.handwriting.v_image_encoder_blocks import FCN_Encoder, FCN_Encoder_SE
from src.model.handwriting.resnet_50_encoder import ResNet50_2layer
from src.model.handwriting.stackmix_image_encoder import get_resnet34_backbone, BiLSTM


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)]
    ])[activation]


def conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, stride):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, stride),
            activation_func(activation),
            conv_bn(self.out_channels, self.out_channels, 1)
        )

        self.activate = activation_func(activation)

        if self.should_apply_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels))
        else:
            self.shortcut = None

    def forward(self, x):

        residual = x

        if self.should_apply_shortcut:
            residual = self.shortcut(x)

        x = self.blocks(x)

        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, n, activation, block, stride):
        super().__init__()

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, activation, stride),
            *[block(out_channels, out_channels, activation, 1) for _ in range(n-1)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, channels, strides, depths, activation, block):

        super().__init__()

        self.channels = channels

        self.input_gate = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            activation_func(activation))

        self.in_out_channel_sizes = list(zip(channels, channels[1:]))

        self.blocks = nn.ModuleList([
            *[ResNetLayer(in_c, out_c, n, activation, block, stride)
              for (in_c, out_c), n, stride in zip(self.in_out_channel_sizes, depths, strides)]
        ])

    def forward(self, x):

        x = self.input_gate(x)

        for block in self.blocks:
            x = block(x)

        return x

class ImgCharModel:
    def __init__(self):
        self.char2index = {}
        self.index2char = {}
        self.n_chars = 2

        self.char2index['ISOS'] = 0
        self.char2index['IEOS'] = 1
        
        self.index2char[0] = 'ISOS'
        self.index2char[1] = 'IEOS'



class StackMixImageEncoder(nn.Module):
    def __init__(self, pos_encoding, config, device, vocab_size):
        super(StackMixImageEncoder, self).__init__()
        self.pos_encoding = pos_encoding
        self.resnet = get_resnet34_backbone(pretrained=True)
        self.device = device
        img_char_model = ImgCharModel()

        sos_token = img_char_model.char2index['ISOS']
        self.sos_token = torch.LongTensor([[sos_token]])
        eos_token = img_char_model.char2index['IEOS']
        self.eos_token = torch.LongTensor([[eos_token]])
       
        self.img_embedding = nn.Embedding(img_char_model.n_chars, config.char_embedding_dim)

        self.bilstm = BiLSTM(256, 256, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (config.char_embedding_dim, config.char_embedding_dim))
        self.classifier = nn.Sequential(
            nn.Linear(config.char_embedding_dim*2, config.char_embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.char_embedding_dim, vocab_size+1)
        )

    def forward(self, src, **kwargs):
        # print("Input:", src.shape)
        aux_ctc = kwargs.pop("aux_ctc", False)
        x = self.resnet(src)
        # print("Resnet:", x.shape)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        # print("Before pooling:", x.shape)
        x = self.avg_pool(x)
        # print("After pooling:", x.shape)
        x = x.transpose(1, 2)
        # print("After transpose:", x.shape)

        bs = src.shape[0]
        sos_token = self.img_embedding(self.sos_token.to(self.device))
        sos_token = sos_token.repeat(bs, 1, 1)
        eos_token = self.img_embedding(self.eos_token.to(self.device))
        eos_token = eos_token.repeat(bs, 1, 1)
        x = torch.cat([sos_token, x, eos_token], axis=1)

        char_embedding =(x.clone() + self.pos_encoding(x))
        
        if aux_ctc:
           # print("After encoding:", x.shape)
           x = self.bilstm(x)
           aux_features = self.classifier(x)
           # print("Aux features:", aux_features.shape)
           aux_features = aux_features.permute(1,0,2).contiguous().log_softmax(2)
           # print("Aux features log softmax:", aux_features.shape)
           return char_embedding, aux_features


        return char_embedding, None

     


class ImageEncoder(nn.Module):

    def __init__(self, pos_encoding, config, device, vocab_size):
        super(ImageEncoder, self).__init__()
        # """

        self.pos_encoding = pos_encoding
        # self.dropout = nn.Dropout(p=config.transformer_encoder['dropout'])

        # block = ResNetBasicBlock

        # self.resnet = ResNet(config.resnet_encoder['in_channels'],
        #                      config.resnet_encoder['channels'],
        #                      config.resnet_encoder['strides'],
        #                      config.resnet_encoder['depths'],
        #                      config.resnet_encoder['activation'],
        #                      block)

        # self.resnet = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] 
        #                             + list(models.resnet50(pretrained=True).children())[1:6])
        #                             + [nn.Conv2d(512, 512, kernel_size=(8,1), stride=1, padding=0, bias=True)])#, nn.BatchNorm2d(512), nn.ReLU(True)])

        # self.resnet = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] 
        #                         +list(models.resnet50(pretrained=False).children())[1:6])
        #                         + [nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(8,1), stride=1, padding=0, groups=512, bias=False),
        #                             nn.BatchNorm2d(512),
        #                             nn.ReLU(True),
        #                             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False),
        #                             nn.BatchNorm2d(512),
        #                             nn.ReLU(True),]
        #                     )



        # params = {"dropout": 0.5, "input_channels": 3}
        params = {"dropout": 0.5, "input_channels": 1}
        # self.resnet = FCN_Encoder(params)
        # self.resnet = FCN_Encoder_SE(params)
        # self.resnet = get_resnet34_backbone(pretrained=True)

        # self.resnet = ResNet50_2layer(small_additional_layer=True, freeze_res_block=False, pretrained_res_blocks=True)


        # encoder_layer = nn.TransformerEncoderLayer(d_model=config.char_embedding_dim,
        #                                            nhead=config.transformer_encoder['num_heads'])

        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
        #                                                  num_layers=config.transformer_encoder['num_layers'])
        # """
        self.device = device
        img_char_model = ImgCharModel()

        sos_token = img_char_model.char2index['ISOS']
        self.sos_token = torch.LongTensor([[sos_token]])
        eos_token = img_char_model.char2index['IEOS']
        self.eos_token = torch.LongTensor([[eos_token]])
       
        self.img_embedding = nn.Embedding(img_char_model.n_chars, config.char_embedding_dim)

        # self.linear_projector = nn.Linear(256, 512)
        self.aux_linear = nn.Linear(config.char_embedding_dim, vocab_size+1)
        # self.avg_pool = nn.AdaptiveAvgPool2d(
        #     (config.char_embedding_dim, config.char_embedding_dim))
        # self.aux_linear = nn.Sequential(
        #     nn.Linear(config.char_embedding_dim, config.char_embedding_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(config.char_embedding_dim, vocab_size+1)
        # )


    def forward(self, src, **kwargs):
        aux_ctc = kwargs.pop("aux_ctc", False)

        # """
        # print("SRC:", src.shape)

        # char_embedding = self.resnet(src)

        char_embedding = src


        # print("Char embedding:", char_embedding.shape)
        # char_embedding = char_embedding.squeeze(dim=-2).permute(0, 2, 1)


        # b, c, h, w = char_embedding.size()
        # char_embedding = char_embedding.view(b, c * h, w)
        # char_embedding = self.avg_pool(char_embedding)
        # char_embedding = src + self.pos_encoding(src)
        # print("Char embedding:", char_embedding.shape)

        # char_embedding = self.dropout(char_embedding + self.pos_encoding(char_embedding))
        # char_embedding = char_embedding.permute(1, 0, 2)

        # char_embedding = self.transformer_encoder(char_embedding)
        # char_embedding = char_embedding.permute(1, 0, 2)
        # char_embedding = nn.functional.normalize(char_embedding, p=2, dim=-1)
        # """
        # char_embedding = f.unfold(src, kernel_size=(32,8), stride=8)
        # char_embedding = char_embedding.permute(0, 2, 1)
        # print(char_embedding.shape)

        # char_embedding = src
        # char_embedding = self.linear_projector(char_embedding)
        # char_embedding = char_embedding + self.pos_encoding(char_embedding)

        bs = src.shape[0]
        sos_token = self.img_embedding(self.sos_token.to(self.device))
        sos_token = sos_token.repeat(bs, 1, 1)
        eos_token = self.img_embedding(self.eos_token.to(self.device))
        eos_token = eos_token.repeat(bs, 1, 1)
        char_embedding = torch.cat([sos_token, char_embedding, eos_token], axis=1)
        char_embedding =(char_embedding + self.pos_encoding(char_embedding))
        # char_embedding_pe =(char_embedding + self.pos_encoding(char_embedding))
        
        if aux_ctc:
           aux_features = self.aux_linear(char_embedding)
           aux_features = aux_features.permute(1,0,2).contiguous().log_softmax(2)
           return char_embedding, aux_features


        #"""

        # return char_embedding, None
        return char_embedding, None
