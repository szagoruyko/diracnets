from functools import partial
from nested_dict import nested_dict
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import dirac, kaiming_normal
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather


def dirac_delta(ni, no, k):
    n = min(ni, no)
    return dirac(torch.Tensor(n, n, k, k)).repeat(max(no // ni, 1), max(ni // no, 1), 1, 1)


def ncrelu(x):
    return torch.cat([x.clamp(min=0),
                      x.clamp(max=0)], dim=1)


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1, gain=1.0):
    return cast(torch.Tensor(no, ni * 2, k, k).normal_(std=gain))


def linear_params(ni, no):
    return cast({'weight': kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)})


def bnparams(n):
    return cast({'weight': torch.rand(n), 'bias': torch.zeros(n)})


def bnstats(n):
    return cast({'running_mean': torch.zeros(n), 'running_var': torch.ones(n)})


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
    return OrderedDict(('.'.join(k), Variable(v, requires_grad=True))
                       for k, v in nested_dict(params).iteritems_flat() if v is not None)


def flatten_stats(stats):
    return OrderedDict(('.'.join(k), v)
                       for k, v in nested_dict(stats).iteritems_flat())


def batch_norm(x, params, stats, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=stats[base + '.running_mean'],
                        running_var=stats[base + '.running_var'],
                        training=mode)


def size2name(size):
    return 'eye' + '_'.join(map(str, size))


def block(o, params, stats, base, mode, j):
    w = params[base + '.conv']
    alpha = params[base + '.alpha'].expand_as(w)
    beta = params[base + '.beta'].expand_as(w)
    delta = Variable(stats[size2name(w.size())])
    w = beta * F.normalize(w.view(w.size(0), -1)).view_as(w) + alpha * delta
    o = F.conv2d(ncrelu(o), w, stride=1, padding=1)
    o = batch_norm(o, params, stats, base + '.bn', mode)
    return o


def group(o, params, stats, base, mode, count):
    for i in range(count):
        o = block(o, params, stats, '%s.block%d' % (base, i), mode, i)
    return o


def define_diracnet(depth, width, dataset):

    def gen_group_params(ni, no, count):
        return {'block%d' % i: {'conv': conv_params(ni if i == 0 else no, no, k=3, gain=0.1),
                                'alpha': cast(torch.Tensor([5])),
                                'beta': cast(torch.Tensor([1e-3])),
                                'bn': bnparams(no)} for i in range(count)}

    def gen_group_stats(no, count):
        return {'block%d' % i: {'bn': bnstats(no)} for i in range(count)}

    if dataset.startswith('CIFAR'):
        n = (depth - 4) // 6
        widths = torch.Tensor([16, 32, 64]).mul(width).int()

        def f(inputs, params, stats, mode):
            o = F.conv2d(inputs, params['conv'], padding=1)
            o = F.relu(batch_norm(o, params, stats, 'bn', mode))
            o = group(o, params, stats, 'group0', mode, n * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, stats, 'group1', mode, n * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, stats, 'group2', mode, n * 2)
            o = F.avg_pool2d(o, 8)
            o = F.linear(o.view(o.size(0), -1), params['fc.weight'], params['fc.bias'])
            return o

        params = {
            'conv': cast(kaiming_normal(torch.Tensor(widths[0], 3, 3, 3))),
            'bn': bnparams(widths[0]),
            'group0': gen_group_params(widths[0], widths[0], n * 2),
            'group1': gen_group_params(widths[0], widths[1], n * 2),
            'group2': gen_group_params(widths[1], widths[2], n * 2),
            'fc': linear_params(widths[2], 10 if dataset == 'CIFAR10' else 100),
        }

        stats = {'group%d' % i: gen_group_stats(no, n * 2)
                 for i, no in enumerate(widths)}
        stats['bn'] = bnstats(widths[0])

    elif dataset == 'ImageNet':
        definitions = {18: [2, 2, 2, 2], 34: [3, 4, 6, 5]}
        widths = torch.Tensor([64, 128, 256, 512]).mul(width).int()
        blocks = definitions[depth]

        def f(inputs, params, stats, mode):
            o = F.conv2d(inputs, params['conv'], padding=3, stride=2)
            o = batch_norm(o, params, stats, 'bn', mode)
            o = F.max_pool2d(o, 3, 2, 1)
            o = group(o, params, stats, 'group0', mode, blocks[0] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, stats, 'group1', mode, blocks[1] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, stats, 'group2', mode, blocks[2] * 2)
            o = F.max_pool2d(o, 2)
            o = group(o, params, stats, 'group3', mode, blocks[3] * 2)
            o = F.avg_pool2d(o, o.size(-1))
            o = F.linear(o.view(o.size(0), -1), params['fc.weight'], params['fc.bias'])
            return o

        params = {
            'conv': cast(kaiming_normal(torch.Tensor(widths[0], 3, 7, 7))),
            'group0': gen_group_params(widths[0], widths[0], 2 * blocks[0]),
            'group1': gen_group_params(widths[0], widths[1], 2 * blocks[1]),
            'group2': gen_group_params(widths[1], widths[2], 2 * blocks[2]),
            'group3': gen_group_params(widths[2], widths[3], 2 * blocks[3]),
            'bn': bnparams(widths[0]),
            'fc': linear_params(widths[-1], 1000),
        }

        stats = {'group%d' % i: gen_group_stats(no, 2 * b)
                 for i, (no, b) in enumerate(zip(widths, blocks))}
        stats['bn'] = bnstats(widths[0])
    else:
        raise ValueError('dataset not understood')

    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    for k, v in flat_params.items():
        if k.find('.conv') > -1:
            no, ni, kh, kw = v.size()
            # to optimize for memory we keep only one dirac-tensor per size
            flat_stats[size2name(v.size())] = cast(dirac_delta(ni, no, kh))

    return f, flat_params, flat_stats
