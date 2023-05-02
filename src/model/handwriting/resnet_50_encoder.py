from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision.models import resnet50


def weights_init(model: Module):
    """Initialize weights for nn Model

    Parameters
    ----------
    model : Module
        torch nn Model
    """
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class ResNet50_2layer(nn.Module):
    def __init__(self, small_additional_layer=True, freeze_res_block=True, pretrained_res_blocks=True):
        super(ResNet50_2layer, self).__init__()
        self.res_block = get_2_layer_resnet50(freeze_layer=freeze_res_block, pretrained=pretrained_res_blocks)
        self.additional_block = get_additional_block(small=small_additional_layer)

        self.additional_block.apply(weights_init)

    def forward(self, x):
        x = self.res_block(x)
        x = self.additional_block(x)

        return x


def get_additional_block(small=True):
    if small:
        additional_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(
                2, 1), padding=(0, 1)),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=512,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(
                2, 1), padding=(0, 1)),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=2,
                stride=1,
                padding=0,
                groups=512,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
    else:
        additional_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(
                2, 1), padding=(0, 1)),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(
                2, 1), padding=(0, 1)),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

    return additional_layer


def get_2_layer_resnet50(freeze_layer=True, pretrained=True):
    # load pretrained resnet50 model
    if pretrained:
        print("Loading pretrained res blocks")
    else:
        print("Loading random init res blocks")

    resnet50_model = resnet50(pretrained=pretrained)
    # create a copy and replace first CNN layer to read single channel image

    resnet_50_1_chnl = deepcopy(resnet50_model)
    resnet_50_1_chnl.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # modify the first layer weight by taking average across 3 dimensions
    r50_state = resnet50_model.state_dict()
    new_conv1_weight = r50_state["conv1.weight"].mean(dim=1).unsqueeze(1)
    r50_state_1_chnl = deepcopy(r50_state)

    r50_state_1_chnl["conv1.weight"] = new_conv1_weight
    resnet_50_1_chnl.load_state_dict(r50_state_1_chnl)

    # extract first two blocks
    r50_2_layers = nn.Sequential(*list(resnet_50_1_chnl.children())[:-4])

    if freeze_layer:
        for parameter in r50_2_layers.parameters():
            parameter.requires_grad = False

    return r50_2_layers
