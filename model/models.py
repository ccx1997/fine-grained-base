import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d and nn.BatchNorm2d and nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            try:
                m.bias.data.zero_()
            except:
                pass


class VGG19(nn.Module):
    def __init__(self, nc, pretrain=True):
        super(VGG19, self).__init__()
        self.features = tv.models.vgg19_bn(pretrained=pretrain).features
        self.classifier = nn.Sequential(nn.Linear(512, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(512, nc),
                                        )
        modules = self.classifier.modules() if pretrain else self.modules()
        initialize(modules)

    def forward(self, x):
        x = self.features(x)    # [b, c, h, w]
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, nc, pretrain=True):
        super(ResNet50, self).__init__()
        self.net = tv.models.resnet50(pretrained=pretrain)
        self.net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.net.fc = nn.Linear(2048, nc)
        modules = self.net.fc.modules() if pretrain else self.net.modules()
        initialize(modules)

    def forward(self, x, cam=False):
        x = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)  # The last feature map
        y = self.net.avgpool(x).squeeze(3).squeeze(2)
        y = self.net.fc(y)
        if cam:
            return y, x
        else:
            return y
        # return self.net(x)
    
    def get_cam(self, feature_map):
        b, c, h, w = feature_map.size()
        x_flatten = feature_map.permute(0, 2, 3, 1).contiguous().view(-1, c)
        response = self.net.fc(x_flatten)
        response = response.contiguous().view(b, h*w, -1)

        # A good normalization way to locate the attentional area
        # res_min, _ = response.min(1, keepdim=True)
        # res_max, _ = response.max(1, keepdim=True)
        # response = (response - res_min) / (res_max - res_min)

        # A normalization way to indicate the effect of other classes
        response = torch.softmax(response, dim=2)

        response = response.contiguous().view(b, h, w, -1).permute(0, 3, 1, 2)
        return response


class AttentionBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(AttentionBlock, self).__init__()
        self.key_gen = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.ReLU())
        self.query_gen = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.ReLU())
        self.value_gen = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.ReLU())
        self.embedding = nn.Sequential(nn.Conv2d(hidden_features, in_features, 1), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(2)

    def softmax_hw(self, x):
        b, d, h, w = x.size()
        return self.softmax(x.view(b, d, -1)).view(b, d, h, w)

    def forward(self, x):
        """
        :param x: [b, c, h, w]
        :return:
        """
        key = self.key_gen(x)  # [b, d, h, w]
        query = self.query_gen(x)  # [b, d, h, w]
        value = self.value_gen(x)  # [b, d, h, w]
        query = self.gap(query)  # [b, d, 1, 1]
        attention = key * query  # [b, d, h, w]
        attention = attention.sum(1, keepdim=True)  # [b, 1, h, w]
        attention = self.softmax_hw(attention)
        context = value * attention
        context = context.sum((2, 3), keepdim=True)  # [b, d, 1, 1]
        context = self.embedding(context)
        x = x + context
        return x, attention


class GeometryAnalytic(nn.Module):
    def __init__(self):
        super(GeometryAnalytic, self).__init__()

    def forward(self, *x):
        pass


if __name__ == "__main__":
    x = torch.rand(2, 3, 448, 448)
    net = ResNet50(20)
    y = net(x)
    print(y.size())
