import unittest
import torch
from diracconv import DiracConv1d, DiracConv2d, DiracConv3d
from torch.autograd import Variable


class TestDirac(unittest.TestCase):

    def test_dirac_property1d(self):
        ni, no, k, pad = 4, 4, 3, 1
        module = DiracConv1d(in_channels=ni, out_channels=no, kernel_size=k, padding=pad, bias=False)
        module.alpha.data.fill_(1)
        module.beta.data.fill_(0)
        x = Variable(torch.randn(4, ni, 5))
        y = module(x)
        self.assertEqual(y.size(), x.size(), 'shape check')
        self.assertEqual((y - x).data.abs().sum(), 0, 'dirac delta property check')

    def test_dirac_property2d(self):
        ni, no, k, pad = 4, 4, 3, 1
        module = DiracConv2d(in_channels=ni, out_channels=no, kernel_size=k, padding=pad, bias=False)
        module.alpha.data.fill_(1)
        module.beta.data.fill_(0)
        x = Variable(torch.randn(4, ni, 5, 5))
        y = module(x)
        self.assertEqual(y.size(), x.size(), 'shape check')
        self.assertEqual((y - x).data.abs().sum(), 0, 'dirac delta property check')

    def test_dirac_property3d(self):
        ni, no, k, pad = 4, 4, 3, 1
        module = DiracConv3d(in_channels=ni, out_channels=no, kernel_size=k, padding=pad, bias=False)
        module.alpha.data.fill_(1)
        module.beta.data.fill_(0)
        x = Variable(torch.randn(4, ni, 5, 5, 5))
        y = module(x)
        self.assertEqual(y.size(), x.size(), 'shape check')
        self.assertEqual((y - x).data.abs().sum(), 0, 'dirac delta property check')

    def test_nonsquare(self):
        ni, no, k, pad = 8, 4, 3, 1
        module = DiracConv2d(in_channels=ni, out_channels=no, kernel_size=k, padding=pad, bias=False)
        x = Variable(torch.randn(4, ni, 5, 5))
        y = module(x)


if __name__ == '__main__':
    unittest.main()
