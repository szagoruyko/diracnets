import unittest
import torch
from diracconv import DiracConv2d
from torch.autograd import Variable


class TestDirac(unittest.TestCase):

    def test_module(self):
        ni, no, k, pad = 4, 4, 3, 1
        module = DiracConv2d(in_channels=ni, out_channels=no, kernel_size=k, padding=pad, bias=False)
        module.alpha.data.fill_(1)
        module.beta.data.fill_(0)
        x = Variable(torch.randn(4, ni, 5, 5))
        y = module(x)
        self.assertEqual(y.size(), x.size())
        self.assertEqual((y - x).data.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
