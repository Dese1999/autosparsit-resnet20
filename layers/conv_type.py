import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from configs.base_config import args as parser_args

DenseConv = nn.Conv2d


# Not learning weights, finding subnet
class SplitConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        mask = kwargs.pop('mask', None)
        self.split_mode = kwargs.pop('split_mode', None)
        self.split_rate = kwargs.pop('split_rate', None)
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        super().__init__(*args, **kwargs)

        # Checking the dimensions of the weights
        #print(f"SplitConv initialized with weight shape: {self.weight.shape}, kernel_size: {self.kernel_size}")

        if self.split_mode == 'kels':
            if self.in_channels_order is None:
                mask = np.zeros((self.weight.size()))
                if self.weight.size()[1] == 3:  # This is the first conv
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :, :, :] = 1
                else:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :math.ceil(self.weight.size()[1] * self.split_rate), :, :] = 1
            else:
                mask = np.zeros((self.weight.size()))
                conv_concat = [int(chs) for chs in self.in_channels_order.split(',')]
                start_ch = 0
                for conv in conv_concat:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), start_ch:start_ch + math.ceil(conv * self.split_rate), :, :] = 1
                    start_ch += conv
        elif self.split_mode == 'wels':
            if mask is None:
                mask = np.random.rand(*list(self.weight.shape))
                threshold = 1 - self.split_rate
                mask[mask < threshold] = 0
                mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2, f'Something is wrong with the mask {np.unique(mask)}'
        else:
            raise NotImplemented(f'Invalid split_mode {self.split_mode}')

        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)


    def extract_slim(self,dst_m,src_name,dst_name):
        c_out, c_in, _, _, = self.weight.size()
        d_out, d_in, _, _ = dst_m.weight.size()
        if self.in_channels_order is None:
            if c_in == 3:
                selected_convs = self.weight[:d_out]
                # is_first_conv = False
            else:
                selected_convs = self.weight[:d_out][:, :d_in, :, :]

            assert selected_convs.shape == dst_m.weight.shape
            dst_m.weight.data = selected_convs
        else:
            selected_convs = self.weight[:d_out, self.mask[0, :, 0, 0] == 1, :, :]
            assert selected_convs.shape == dst_m.weight.shape, '{} {} {} {}'.format(dst_name, src_name, dst_m.weight.shape,
                                                                                    selected_convs.shape)
            dst_m.weight.data = selected_convs

    def reset_mask(self):
        if self.split_mode == 'wels':
            mask = np.random.rand(*list(self.weight.shape))
            threshold = 1 - self.split_rate
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2,'Something is wrong with the score {}'.format(np.unique(mask))
        else:
            raise NotImplemented('Reset score randomly only with WELS. The current mode is '.format(self.split_mode))
        # scores = np.zeros((self.weight.size()))
        # rand_sub = random.randint(0, self.weight.size()[0] - math.ceil(self.weight.size()[0] * self.keep_rate))
        # if self.weight.size()[1] == 3:  ## This is the first conv
        #     scores[rand_sub:rand_sub+math.ceil(self.weight.size()[0] * self.keep_rate), :, :, :] = 1
        # else:
        #     scores[rand_sub:rand_sub+math.ceil(self.weight.size()[0] * self.keep_rate), :math.ceil(self.weight.size()[1] * self.keep_rate), :,
        #     :] = 1
    
        self.mask.data = torch.Tensor(mask).cuda()
        # raise NotImplemented('Not implemented yet')
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    # def reset_bias_scores(self):
    #     pass

    # def set_split_rate(self, split_rate, bias_split_rate):
    #     self.split_rate = split_rate
    #     if self.bias is not None:
    #         self.bias_split_rate = bias_split_rate
    #     else:
    #         self.bias_split_rate = 1.0

    def split_reinitialize(self,cfg):
        if cfg.evolve_mode == 'rand':
            rand_tensor = torch.zeros_like(self.weight).cuda()
            nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
            self.weight.data = torch.where(self.mask.type(torch.bool), self.weight.data, rand_tensor)
        elif cfg.evolve_mode == 'zero':
            rand_tensor = torch.zeros_like(self.weight).cuda()
            self.weight.data = torch.where(self.mask.type(torch.bool), self.weight.data, rand_tensor)
        else:
            raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))

        if hasattr(self, "bias") and self.bias is not None and self.bias_split_rate < 1.0:
            bias_mask = self.mask[:, 0, 0, 0]  ## Same conv mask is used for bias terms
            if cfg.evolve_mode == 'rand':
                rand_tensor = torch.zeros_like(self.bias)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(rand_tensor, -bound, bound)
                self.bias.data = torch.where(bias_mask.type(torch.bool), self.bias.data, rand_tensor)
            elif cfg.evolve_mode == 'zero':
                rand_tensor = torch.zeros_like(self.bias)
                self.bias.data = torch.where(bias_mask.type(torch.bool), self.bias.data, rand_tensor)
            else:
                raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))

    def forward(self, x):
        ## Debugging reasons only
        # if self.split_rate < 1:
        #     w = self.mask * self.weight
        #     if self.bias_split_rate < 1:
        #         # bias_subnet = GetSubnet.apply(self.clamped_bias_scores, self.bias_keep_rate)
        #         b = self.bias * self.mask[:, 0, 0, 0]
        #     else:
        #         b = self.bias
        # else:
        #     w = self.weight
        #     b = self.bias

        w = self.weight
        b = self.bias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x
