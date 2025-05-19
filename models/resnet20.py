# resnet20.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import layers

class Block(nn.Module):
    """A ResNet block."""
    def __init__(self, f_in: int, f_out: int, downsample=False):
        super(Block, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = layers.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layers.BatchNorm2d(f_out)
        self.conv2 = layers.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layers.BatchNorm2d(f_out)
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                layers.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                layers.BatchNorm2d(f_out)
            )
        else:
            self.shortcut = layers.Identity2d(f_in)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""
    def __init__(self, plan, num_classes, dense_classifier):
        super(ResNet, self).__init__()
        current_filters = plan[0][0]
        self.conv = layers.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = layers.BatchNorm2d(current_filters)
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters
        self.blocks = nn.Sequential(*blocks)
        self.fc = layers.Linear(plan[-1][0], num_classes) if dense_classifier else nn.Linear(plan[-1][0], num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Added methods for compatibility with KE and pruning
    def get_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def set_params(self, new_params):
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in self.parameters():
            cand_params = new_params[progress: progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self):
        grads = []
        for pp in self.parameters():
            if pp.grad is None:
                grads.append(torch.zeros(pp.shape).view(-1).to(pp.device))
            else:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def get_grads_list(self):
        return [pp.grad.view(-1) if pp.grad is not None else torch.zeros_like(pp).view(-1) for pp in self.parameters()]

def _plan(D, W):
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]
    return plan

def _resnet(arch, plan, num_classes, dense_classifier, pretrained):
    model = ResNet(plan, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 16)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained)