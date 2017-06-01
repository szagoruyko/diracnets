import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import linear_params, bnparams, bnstats, cast,\
        flatten_params, flatten_stats
from .delta import delta, eye

import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
from nested_dict import nested_dict
from collections import OrderedDict

from torch.nn.init import orthogonal, dirac


def init_square(n, k, gain):
    base = torch.zeros(n, n, k, k)
    if k == 1:
        base = orthogonal(base, gain)
    else:
        m = (k - 1) // 2
        base[:,:,m,m] = orthogonal(base[:,:,m,m].contiguous(), gain)
    return base


def delta(ni, no, k=1, gain=1e-3):
    return init_square(min(ni, no), k, gain).repeat(max(no // ni, 1), max(ni // no, 1), 1, 1)


def eye(ni, no, k):
    n = min(ni, no)
    return dirac(torch.Tensor(n, n, k, k)).repeat(max(no // ni, 1), max(ni // no, 1), 1, 1)

def ncrelu(x):
    return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda(), dtype)()


def linear_params(ni,no):
    return cast(dict(
        weight=torch.Tensor(no,ni).normal_(0,2/math.sqrt(ni)),
        bias=torch.zeros(no)))


def bnparams(n):
    return cast(dict(
        weight=torch.Tensor(n).uniform_(),
        bias=torch.zeros(n)))


def bnstats(n):
    return cast(dict(
        running_mean=torch.zeros(n),
        running_var=torch.ones(n)))


def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k,v in param_dict.iteritems():
            for i,u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [partial(f, params=p, stats=s, mode=mode)
                for p,s in zip(params_replicas, stats_replicas)]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten_params(params):
    flat_params = OrderedDict()
    for keys, v in nested_dict(params).iteritems_flat():
        if v is not None:
            flat_params['.'.join(keys)] = Variable(v, requires_grad=True)
    return flat_params


def flatten_stats(stats):
    flat_stats = OrderedDict()
    for keys, v in nested_dict(stats).iteritems_flat():
        flat_stats['.'.join(keys)] = v
    return flat_stats


def ncrelu(x):
    return torch.cat([F.relu(x), -F.relu(-x)], 1)

def conv_params(ni, no, k=1, gain=2.0):
    return cast(delta(ni * 2, no, k, gain))


def nina(depth, width, num_classes, opt):
    params = {}
    n = (depth - 4) // 6
    blocks = torch.Tensor([16, 32, 64]).mul(width).int()

    def gen_block_params(ni, no, gain):
        return {'conv0': conv_params(ni, no, k=3, gain=gain),
                'conv1': conv_params(no, no, k=3, gain=gain),
                'bn0': bnparams(no),
                'bn1': bnparams(no)}

    def gen_block_stats(ni, no):
        return {'bn%d' % i: bnstats(no) for i in range(2)}

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no, opt.gain)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: gen_block_stats(ni if i == 0 else no, no)
                for i in range(count)}

    params = {
        'conv': cast(torch.nn.init.kaiming_normal(torch.Tensor(blocks[0], 3, 3, 3))),
        'group0': gen_group_params(blocks[0], blocks[0], n),
        'group1': gen_group_params(blocks[0], blocks[1], n),
        'group2': gen_group_params(blocks[1], blocks[2], n),
        'fc': linear_params(blocks[2], num_classes),
        'bn': bnparams(blocks[0]),
    }

    stats = {'group%d' % i: gen_group_stats(blocks[max(0, i - 1)], b, n)
             for i, b in enumerate(blocks)}
    stats['bn'] = bnstats(blocks[0])

    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    for k, v in flat_params.items():
        if k.find('.conv') > -1:
            no, ni, kh, kw = v.size()
            I = cast(eye(ni, no, kh))
            flat_stats['eye' + '_'.join(map(str,I.size()))] = I
            flat_params[k + '.gamma'] = Variable(cast(torch.ones(1) * 5), requires_grad=True)
            flat_params[k + '.beta'] = Variable(cast(torch.ones(1) * 1e-3), requires_grad=True)

    def activation(x, params, stats, base, mode):
        return F.batch_norm(x, weight=params[base + '.weight'],
                            bias=params[base + '.bias'],
                            running_mean=stats[base + '.running_mean'],
                            running_var=stats[base + '.running_var'],
                            training=mode)

    def block(o, params, stats, base, mode, j):
        for i in range(2):
            name = '%s.conv%d' % (base, i)
            w = params[name]
            gamma = params[name + '.gamma'].expand_as(w)
            beta = params[name + '.beta'].expand_as(w)
            eye = Variable(stats['eye' + '_'.join(map(str, w.size()))])
            w = beta * F.normalize(w.view(w.size(0), -1)).view_as(w) + gamma * eye
            o = F.conv2d(ncrelu(o), w, stride=1, padding=1)
            o = activation(o, params, stats, '%s.bn%d' % (base,i), mode)
        return o

    def group(o, params, stats, base, mode, count):
        for i in range(count):
            o = block(o, params, stats, '%s.block%d' % (base, i), mode, i)
        return o

    def f(inputs, params, stats, mode):
        o = F.conv2d(inputs, params['conv'], padding=1)
        o = F.relu(activation(o, params, stats, 'bn', mode))
        o = group(o, params, stats, 'group0', mode, n)
        o = F.max_pool2d(o, 2)
        o = group(o, params, stats, 'group1', mode, n)
        o = F.max_pool2d(o, 2)
        o = group(o, params, stats, 'group2', mode, n)
        o = F.avg_pool2d(o, 8)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params, flat_stats
