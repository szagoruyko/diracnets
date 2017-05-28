DiracNets
=========

Code and models for DiracNets: Training Very Deep Neural Networks Without Skip-Connections.

Networks with skip-connections like ResNet show excellent performance in image recognition benchmarks, but do not benefit from increased depth, we are thus still interested in learning __actually__ deep representations, and the benefits they could bring. We propose a simple weight parameterization, which improves training of deep plain (without skip-connections) networks, and allows training plain networks with hundreds of layers. Accuracy of our proposed DiracNets is close to Wide ResNet (although needs more parameters to achieve it), and we are able to outperform ResNet-1000 with plain DiracNet with only 34 layers. Also, the proposed Dirac weight parameterization can be folded into one filter for inference, leading to easily interpretable VGG-like network.


## TL;DR

In a nutshell, Dirac parameterization is simply a sum of filters and Dirac delta function:

```
conv2d(x, delta + W)
```

To plug it into a plain network, we add several learnable scalar parameters and weight normalization.
Here is simplified PyTorch-like pseudocode for the function:

```python
def dirac_conv2d(input, W, alpha, beta)
    return F.conv2d(input, alpha * dirac(W.data) + beta * F.normalize(W))
```

We also use NCReLU (negative CReLU) nonlinearity:

```python
def ncrelu(x):
    return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)
```


## Code

### nn.Module code

We provide `DiracConv1d`, `DiracConv2d`, `DiracConv3d`, which work like `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, but have Dirac-parametrization inside.
Training code doesn't use these modules, and uses only functional interface to PyTorch, `torch.nn.functional`.


## Pretrained models

We fold batch normalization and Dirac parameterization into `F.conv2d` `weight` and `bias` tensors for simplicity. Resulting models are as simple as VGG or AlexNet, having only nonlinearity+conv2d as a basic block.
