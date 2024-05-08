import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None
    ):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(self.bn1, nn.Sequential) and isinstance(self.bn2, nn.Sequential):
            return x
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if (not isinstance(self.conv1, nn.Conv2d)):
                if len(self.conv1) == 2:
                    if self.conv1[0].padding[2] == 0:
                        x = x[:, :, 1:, :]
            identity = self.downsample(x)

        if not (isinstance(self.conv1, nn.Conv2d) and isinstance(self.conv2, nn.Conv2d)):
            if len(self.conv1) == 2 or len(self.conv2) == 2:
                if not (self.conv1[0].padding[2] == 0 and self.conv1[0].padding[3] == 0 and self.conv2[0].padding[2] == 0 and self.conv2[0].padding[3] == 0):
                    if self.conv2[0].padding[3] != 0:
                        identity = identity[:,:,-out.shape[2]:,:]
                    elif self.conv2[0].padding[2] != 0:
                        identity = identity[:,:,:out.shape[2],:]
                    elif self.conv1[0].padding[2] != 0:
                        identity = identity[:,:, 1:out.shape[2]+1,:]
                    elif self.conv1[0].padding[3] != 0:
                        identity = identity[:,:,-out.shape[2]-1:-1,:]

                elif (identity.shape[2] - out.shape[2]) == 4:
                    identity = identity[:,:,2:-2,:]
                elif identity.shape != out.shape:
                    identity = identity[:,:,1:out.shape[2]+1,:]
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(self.bn1, nn.Sequential) and isinstance(self.bn2, nn.Sequential) and isinstance(self.bn3, nn.Sequential):
            return x
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if (not isinstance(self.conv2, nn.Conv2d)):
                if len(self.conv2) == 2:
                    if self.conv2[0].padding[2] == 0 and self.downsample[0].stride[0] != 1:
                        x = x[:, :, 1:, :]
            identity = self.downsample(x)

        if not (isinstance(self.conv1, nn.Conv2d) and isinstance(self.conv2, nn.Conv2d)):
            if len(self.conv2) == 2:
                if not (self.conv2[0].padding[2] == 0 and self.conv2[0].padding[3] == 0):
                    if self.conv2[0].padding[3] != 0:
                        identity = identity[:,:,-out.shape[2]:,:]
                    elif self.conv2[0].padding[2] != 0:
                        identity = identity[:,:,:out.shape[2],:]

                elif (identity.shape[2] - out.shape[2]) == 2:
                    identity = identity[:,:,1:-1,:]

                elif identity.shape != out.shape:
                    identity = identity[:,:,1:out.shape[2]+1,:]

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet34(pretrained = False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet34-b627a593.pth", model_dir="./model_data", map_location='cpu')
        model.load_state_dict(state_dict)

    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features    = nn.Sequential(*features)

    return features

def resnet50(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features    = nn.Sequential(*features)

    return features

def resnet101(pretrained = False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet101-63fe2227.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features    = nn.Sequential(*features)

    return features

def resnet152(pretrained = False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])

    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features    = nn.Sequential(*features)

    return features