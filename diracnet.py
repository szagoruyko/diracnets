from functools import partial
from nested_dict import nested_dict
import torch
import torch.nn.functional as F
from torch.nn.init import dirac_, kaiming_normal_
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from torch import nn
from torch.utils import model_zoo


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True


def size2name(size):
    return 'eye' + '_'.join(map(str, size))


def block(o, params, base, mode, j):
    w = params[base + '.conv']
    alpha = params[base + '.alpha'].view(-1, 1, 1, 1)
    beta = params[base + '.beta'].view(-1, 1, 1, 1)
    delta = params[size2name(w.shape)]
    w = beta * F.normalize(w.view(w.shape[0], -1)).view_as(w) + alpha * delta
    o = F.conv2d(F.relu(o), w, stride=1, padding=1)
    o = batch_norm(o, params, base + '.bn', mode)
    return o


def group(o, params, base, mode, count):
    for i in range(count):
        o = block(o, params, '%s.block%d' % (base, i), mode, i)
    return o


def define_diracnet(depth, width, dataset):

    def gen_group_params(ni, no, count):
        return {'block%d' % i: {'conv': conv_params(ni if i == 0 else no, no, k=3),
                                'alpha': cast(torch.ones(no).fill_(1)),
                                'beta': cast(torch.ones(no).fill_(0.1)),
                                'bn': bnparams(no)} for i in range(count)}

    if dataset.startswith('CIFAR'):
        n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]

        def f(inputs, params, mode):
            o = F.conv2d(inputs, params['conv'], padding=1)
            o = F.relu(batch_norm(o, params, 'bn', mode))
            o = group(o, params, 'group0', mode, n * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, 'group1', mode, n * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, 'group2', mode, n * 2)
            o = F.avg_pool2d(F.relu(o), 8)
            o = F.linear(o.view(o.size(0), -1), params['fc.weight'], params['fc.bias'])
            return o

        params = {
            'conv': cast(kaiming_normal_(torch.Tensor(widths[0], 3, 3, 3))),
            'bn': bnparams(widths[0]),
            'group0': gen_group_params(widths[0], widths[0], n * 2),
            'group1': gen_group_params(widths[0], widths[1], n * 2),
            'group2': gen_group_params(widths[1], widths[2], n * 2),
            'fc': linear_params(widths[2], 10 if dataset == 'CIFAR10' else 100),
        }

    elif dataset == 'ImageNet':
        definitions = {18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3]}
        widths = [int(width * v) for v in (64, 128, 256, 512)]
        blocks = definitions[depth]

        def f(inputs, params, mode):
            o = F.conv2d(inputs, params['conv'], padding=3, stride=2)
            o = batch_norm(o, params, 'bn', mode)
            o = F.max_pool2d(o, 3, 2, 1)
            o = group(o, params, 'group0', mode, blocks[0] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, 'group1', mode, blocks[1] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, 'group2', mode, blocks[2] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, 'group3', mode, blocks[3] * 2)
            o = F.avg_pool2d(F.relu(o), o.size(-1))
            o = F.linear(o.view(o.size(0), -1), params['fc.weight'], params['fc.bias'])
            return o

        params = {
            'conv': cast(kaiming_normal_(torch.Tensor(widths[0], 3, 7, 7))),
            'group0': gen_group_params(widths[0], widths[0], 2 * blocks[0]),
            'group1': gen_group_params(widths[0], widths[1], 2 * blocks[1]),
            'group2': gen_group_params(widths[1], widths[2], 2 * blocks[2]),
            'group3': gen_group_params(widths[2], widths[3], 2 * blocks[3]),
            'bn': bnparams(widths[0]),
            'fc': linear_params(widths[-1], 1000),
        }
    else:
        raise ValueError('dataset not understood')

    flat_params = flatten(params)

    set_requires_grad_except_bn_(flat_params)

    for k, v in list(flat_params.items()):
        if k.find('.conv') > -1:
            flat_params[size2name(v.size())] = cast(dirac_(v.data.clone()))

    return f, flat_params


model_urls = {
    'diracnet18': 'https://s3.amazonaws.com/modelzoo-networks/diracnet18v2folded-a2174e15.pth',
    'diracnet34': 'https://s3.amazonaws.com/modelzoo-networks/diracnet34v2folded-dfb15d34.pth'
}


class DiracNet(nn.Module):

    widths = (64, 128, 256, 512)
    block_depths = {18: [v * 2 for v in (2, 2, 2, 2)],
                    34: [v * 2 for v in (3, 4, 6, 3)]}

    def __init__(self, depth=18):
        super().__init__()
        self.features = nn.Sequential()
        n_channels = self.widths[0]
        self.features.add_module('conv', nn.Conv2d(3, n_channels, kernel_size=7, stride=2, padding=3))
        self.features.add_module('max_pool0', nn.MaxPool2d(3, 2, 1))
        for group_id, (width, block_depth) in enumerate(zip(self.widths, self.block_depths[depth])):
            for block_id in range(block_depth):
                name = 'group{}.block{}.'.format(group_id, block_id)
                self.features.add_module(name + 'relu', nn.ReLU())
                self.features.add_module(name + 'conv', nn.Conv2d(n_channels, width, kernel_size=3, padding=1))
                n_channels = width
            if group_id != 3:
                self.features.add_module('max_pool{}'.format(group_id + 1), nn.MaxPool2d(2))
            else:
                self.features.add_module('last_relu', nn.ReLU())
                self.features.add_module('avg_pool', nn.AvgPool2d(7))
        self.fc = nn.Linear(in_features=512, out_features=1000)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def diracnet18(pretrained=False):
    model = DiracNet(18)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['diracnet18']))
    return model


def diracnet34(pretrained=False):
    model = DiracNet(34)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['diracnet34']))
    return model

