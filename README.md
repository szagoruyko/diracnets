DiracNets
=========

Code and models for DiracNets: Training Very Deep Neural Networks Without Skip-Connections.

Networks with skip-connections like ResNet show excellent performance in image recognition benchmarks, but do not benefit from increased depth, we are thus still interested in learning __actually__ deep representations, and the benefits they could bring. We propose a simple weight parameterization, which improves training of deep plain (without skip-connections) networks, and allows training plain networks with hundreds of layers. Accuracy of our proposed DiracNets is close to Wide ResNet (although DiracNets need more parameters to achieve it), and we are able to outperform ResNet-1000 with plain DiracNet with only 34 layers. Also, the proposed Dirac weight parameterization can be folded into one filter for inference, leading to easily interpretable VGG-like network.

<img src=http://imagine.enpc.fr/~zagoruys/delta-circles.svg>

## TL;DR

In a nutshell, Dirac parameterization is simply a sum of filters and Dirac delta function:

```python
conv2d(x, delta + W)
```

To plug it into a plain network, we add learnable scalar parameters `alpha`, `beta` and weight normalization.
Here is simplified PyTorch-like pseudocode for the function:

```python
def dirac_conv2d(input, W, alpha, beta)
    return F.conv2d(input, alpha * dirac(W) + beta * F.normalize(W))
```

We also use NCReLU (negative CReLU) nonlinearity:

```python
def ncrelu(x):
    return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)
```


## Code

Code structure:

```
├── README.md       # this file
├── diracconv.py    # DiracConv definitions
├── test.py         # unit tests
└── train.py        # CIFAR and ImageNet training code
```

### nn.Module code

We provide `DiracConv1d`, `DiracConv2d`, `DiracConv3d`, which work like `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, but have Dirac-parametrization inside.
Training code doesn't use these modules, and uses only functional interface to PyTorch, `torch.nn.functional`.


## Pretrained models

We fold batch normalization and Dirac parameterization into `F.conv2d` `weight` and `bias` tensors for simplicity. Resulting models are as simple as VGG or AlexNet, having only nonlinearity+conv2d as a basic block.

See [diracnets.ipynb](diracnets.ipynb) for functional and modular model definitions.

We provide printout of DiracNet-18-0.75 sequential model for reference:

```
Sequential (
  (conv): Conv2d(3, 48, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (max_pool0): MaxPool2d (size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1))
  (group0.block0.bn): Affine(48)
  (group0.block0.ncrelu): NCReLU()
  (group0.block0.conv): Conv2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group0.block1.ncrelu): NCReLU()
  (group0.block1.conv): Conv2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group0.block2.ncrelu): NCReLU()
  (group0.block2.conv): Conv2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group0.block3.ncrelu): NCReLU()
  (group0.block3.conv): Conv2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (max_pool1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (group1.block0.bn): Affine(48)
  (group1.block0.ncrelu): NCReLU()
  (group1.block0.conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group1.block1.ncrelu): NCReLU()
  (group1.block1.conv): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group1.block2.ncrelu): NCReLU()
  (group1.block2.conv): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group1.block3.ncrelu): NCReLU()
  (group1.block3.conv): Conv2d(192, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (max_pool2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (group2.block0.bn): Affine(96)
  (group2.block0.ncrelu): NCReLU()
  (group2.block0.conv): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group2.block1.ncrelu): NCReLU()
  (group2.block1.conv): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group2.block2.ncrelu): NCReLU()
  (group2.block2.conv): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group2.block3.ncrelu): NCReLU()
  (group2.block3.conv): Conv2d(384, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (max_pool3): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (group3.block0.bn): Affine(192)
  (group3.block0.ncrelu): NCReLU()
  (group3.block0.conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group3.block1.ncrelu): NCReLU()
  (group3.block1.conv): Conv2d(768, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group3.block2.ncrelu): NCReLU()
  (group3.block2.conv): Conv2d(768, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (group3.block3.ncrelu): NCReLU()
  (group3.block3.conv): Conv2d(768, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU ()
  (avg_pool): AvgPool2d (
  )
  (view): Flatten()
  (fc): Linear (384 -> 1000)
)
```
