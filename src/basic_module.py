import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

logabs = lambda x: torch.log(torch.abs(x) + 1e-6)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class dense_conv(nn.Module):
    def __init__(self, in_feats, grow_rate, kernel=3, activation=nn.ReLU(inplace=True), bias=True):
        super(dense_conv, self).__init__()
        layer = []
        layer.append(nn.Conv2d(in_feats, grow_rate, kernel, padding=1, dilation=1, bias=bias))
        layer.append(activation)
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        output = self.layer(x)
        return torch.cat((x, output), 1)


class dense_block(nn.Module):
    def __init__(self, in_channels, out_channels, grow_rate, n_units, activation, bias=True):
        super(dense_block, self).__init__()
        body = []
        for i in range(n_units):
            body.append(dense_conv(in_channels + i * grow_rate, grow_rate, bias=bias, activation=activation))
        self.body = nn.Sequential(*body)
        self.gate = nn.Conv2d(in_channels + n_units * grow_rate, out_channels, 3, padding=1, bias=bias)

    def forward(self, x):
        y = self.gate(self.body(x))
        return y


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

# Shifting Operation
def get_shift(x, shift=(0, 0)):
    y = torch.roll(x, shifts=shift, dims=(2, 3))
    if shift[0] >= 0:
        y = y[:, :, shift[0]:, :]
        y = torch.nn.functional.pad(y, pad=(0, 0, shift[0], 0), mode='reflect')
    else:
        y = y[:, :, :shift[0], :]
        y = torch.nn.functional.pad(y, pad=(0, 0, 0, -shift[0]), mode='reflect')

    if shift[1] >= 0:
        y = y[:, :, :, shift[1]:]
        y = torch.nn.functional.pad(y, pad=(shift[1], 0, 0, 0), mode='reflect')
    else:
        y = y[:, :, :, :shift[1]]
        y = torch.nn.functional.pad(y, pad=(0, -shift[1], 0, 0), mode='reflect')
    return y


def get_shift_m(x, num=1):
    B, C, W, H = x.size()
    shift = max(math.ceil(np.sqrt(num) / 2), 1)
    cnt = 1
    if cnt >= num:
        return x
    for i in range(-shift, shift + 1):
        for j in range(-shift, shift + 1):
            if i != 0 or j != 0:
                x = torch.cat((x, get_shift(x[:, :C, ], shift=(i, j))), dim=1)
                cnt += 1
            if cnt >= num:
                return x


def weight_inverse(w, type='Inv'):
    w = w.squeeze()
    Q = torch.matmul(w.transpose(1, 0), w)
    if type == 'Inv':  # Calculate the left inverse of the convolutional weights
        # w_inv = torch.matmul(Q.inverse(), w.transpose(1, 0))
        w_inv = torch.matmul(Q.cpu().inverse().to(Q.device), w.transpose(1, 0))
        return w_inv.unsqueeze(-1).unsqueeze(-1)
    elif type == 'Det':  # Log-determinant of solving left inverse
        # w_D = torch.det(Q)
        w_D = torch.det(Q.cpu()).to(Q.device)
        return w_D
    else:
        raise InterruptedError


# InvNN implemented by affine coupling layers, referring to IRN
class Flow(nn.Module):
    def __init__(self, n_channels, split_channels):
        super(Flow, self).__init__()
        self.split_len1 = split_channels
        self.split_len2 = n_channels - split_channels
        self.F = DenseBlock(self.split_len2, self.split_len1, 'xavier', 32, True)
        self.G = DenseBlock(self.split_len1, self.split_len2, 'xavier', 32, True)
        self.H = DenseBlock(self.split_len1, self.split_len2, 'xavier', 32, True)
        self.clamp = 1.

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)
        return torch.cat((y1, y2), 1)

    def jacobian(self, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac


# Generic Invertible Convolution
class WIC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(WIC, self).__init__()
        if in_channels <= out_channels:
            self.iConv2d = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, False)
        elif in_channels > out_channels:
            K = math.ceil(in_channels / out_channels)
            self.iConv2d = nn.Conv2d(in_channels, out_channels * K, 1, 1, 0, 1, 1, False)
        else:
            raise InterruptedError

        self.iConv2d.weight.data = torch.nn.init.orthogonal_(self.iConv2d.weight.data.squeeze()) \
            .unsqueeze(-1).unsqueeze(-1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x, rev=False):
        if self.in_channels <= self.out_channels:
            if not rev:
                if self.training:
                    y = self.iConv2d(x)
                else:
                    y = F.conv2d(x, self.iConv2d.weight[:self.out_channels, ])
            else:
                w_inv = weight_inverse(self.iConv2d.weight, type='Inv')
                if self.training:
                    y = F.conv2d(x, w_inv)
                else:
                    y = F.conv2d(x, w_inv[:self.in_channels, ])

        elif self.in_channels > self.out_channels:
            if not rev:
                if self.training:
                    y = self.iConv2d(x)
                else:
                    y = F.conv2d(x, self.iConv2d.weight[:self.out_channels, ])
            else:
                x = get_shift_m(x, num=math.ceil(self.in_channels / self.out_channels))
                w_inv = weight_inverse(self.iConv2d.weight, type='Inv')
                if self.training:
                    y = F.conv2d(x, w_inv)
                else:
                    y = F.conv2d(x, w_inv[:self.in_channels, ])
        return y

    def jacobian(self):
        # calculate only when not reverse phase
        det = weight_inverse(self.iConv2d.weight, type='Det')
        return torch.mean(torch.abs(torch.log(det + 1e-6)))


# Squeeze Operation for changing spatial resolution
class PixelShuffler(nn.Module):
    def __init__(self, scale=2, downscale=True):
        super(PixelShuffler, self).__init__()
        if downscale:
            self.forw = nn.PixelUnshuffle(scale)
            self.back = nn.PixelShuffle(scale)
        else:
            self.forw = nn.PixelShuffle(scale)
            self.back = nn.PixelUnshuffle(scale)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw(x)
        else:
            return self.back(x)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            # self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            # self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            # self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            # self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    # def jacobian(self):
    #     return self.last_jac

# WICM module with long-term memories
class WICM(nn.Module):
    def __init__(self, n_channels, n_modules, n_blocks, splitting_ratio=0.75):
        super(WICM, self).__init__()

        transform = {}
        feature = {}
        tail = {}

        self.splitting_ratio = splitting_ratio
        self.n_channels = n_channels
        self.n_modules = n_modules
        self.n_blocks = n_blocks

        self.SM_channels = int(n_channels * splitting_ratio)  # short-term memory
        self.LM_channels = n_channels - self.SM_channels  # long-term memory

        for i in range(n_modules):
            # GIC layer
            transform['{}'.format(i)] = WIC(self.SM_channels, n_channels)
            # InvNN
            for j in range(n_blocks):
                feature['{}_{}'.format(i, j)] = Flow(self.SM_channels, self.SM_channels // 2)

        out_channels = self.SM_channels + n_modules * self.LM_channels
        for k in range(1):
            tail['{}'.format(k)] = Flow(out_channels, out_channels // 2)

        self.transform = nn.ModuleDict(transform)
        self.feature = nn.ModuleDict(feature)
        self.tail = nn.ModuleDict(tail)

    def forward(self, x, rev=False):
        y = x
        if not rev:
            jacobian = 0
            int_feat = []
            for i in range(self.n_modules):
                y = self.transform['{}'.format(i)].forward(y, rev)
                if self.training:
                    jacobian += self.transform['{}'.format(i)].jacobian()

                int_feat.append(y.narrow(1, 0, self.LM_channels))
                y = y.narrow(1, self.LM_channels, self.SM_channels)

                for j in range(self.n_blocks):
                    y = self.feature['{}_{}'.format(i, j)].forward(y, rev)
                    # if self.training:
                    #     jacobian += self.feature['{}_{}'.format(i, j)].jacobian()

            int_feat.append(y)
            y = torch.cat(int_feat, dim=1)
            for k in range(len(self.tail)):
                y = self.tail['{}'.format(k)].forward(y, rev)
                # if self.training:
                #     jacobian += self.tail['{}'.format(k)].jacobian()

            if self.training:
                return y, jacobian
            else:
                return y
        else:
            for k in reversed(range(len(self.tail))):
                y = self.tail['{}'.format(k)].forward(y, rev)
            int_feat = []
            for i in range(self.n_modules):
                int_feat.append(y.narrow(1, i*self.LM_channels, self.LM_channels))
            y = y.narrow(1, self.n_modules*self.LM_channels, self.SM_channels)

            for i in reversed(range(self.n_modules)):
                for j in reversed(range(self.n_blocks)):
                    y = self.feature['{}_{}'.format(i, j)].forward(y, rev)
                y = torch.cat((int_feat[i], y), dim=1)
                y = self.transform['{}'.format(i)].forward(y, rev)

            return y