DiracNets
=========

### v2 update (January 2018):

The code was updated for DiracNets-v2 in which we removed NCReLU by adding per-channel `a` and `b` multipliers without weight decay.
This allowed us to significantly simplify the network, which is now folds into a simple chain of convolution-ReLU layers, like VGG.
On ImageNet DiracNet-18 and DiracNet-34 closely match corresponding ResNet with the same number of parameters.

See v1 branch for DiracNet-v1.

-----

PyTorch code and models for *DiracNets: Training Very Deep Neural Networks Without Skip-Connections*

<https://arxiv.org/abs/1706.00388>

Networks with skip-connections like ResNet show excellent performance in image recognition benchmarks, but do not benefit from increased depth, we are thus still interested in learning __actually__ deep representations, and the benefits they could bring. We propose a simple weight parameterization, which improves training of deep plain (without skip-connections) networks, and allows training plain networks with hundreds of layers. Accuracy of our proposed DiracNets is close to Wide ResNet (although DiracNets need more parameters to achieve it), and we are able to match ResNet-1000 accuracy with plain DiracNet with only 28 layers. Also, the proposed Dirac weight parameterization can be folded into one filter for inference, leading to easily interpretable VGG-like network.

DiracNets on ImageNet:
<img src=http://imagine.enpc.fr/~zagoruys/img/diracnet_imagenet.svg>


## TL;DR

In a nutshell, Dirac parameterization is a sum of filters and scaled Dirac delta function:

```
conv2d(x, alpha * delta + W)
```

Here is simplified PyTorch-like pseudocode for the function we use to train plain DiracNets (with weight normalization):

```python
def dirac_conv2d(input, W, alpha, beta)
    return F.conv2d(input, alpha * dirac(W) + beta * normalize(W))
```

where `alpha` and `beta` are per-channel scaling multipliers, and `normalize` does l_2 normalization over each feature plane.


## Code

Code structure:

├── [README.md](README.md)          # this file<br>
├── [diracconv.py](diracconv.py)    # modular DiracConv definitions<br>
├── [test.py](test.py)              # unit tests<br>
├── [diracnet-export.ipynb](diracnet-export.ipynb) # ImageNet pretrained models<br>
├── [diracnet.py](diracnet.py)      # functional model definitions<br>
└── [train.py](train.py)            # CIFAR and ImageNet training code<br>

### Requirements

First install [PyTorch](https://pytorch.org), then install [torchnet](https://github.com/pytorch/tnt):

```
pip install git+https://github.com/pytorch/tnt.git@master
```

Install other Python packages:

```
pip install -r requirements.txt
```

To train DiracNet-34-2 on CIFAR do:

```
python train.py --save ./logs/diracnets_$RANDOM$RANDOM --depth 34 --width 2
```

To train DiracNet-18 on ImageNet do:

```bash
python train.py --dataroot ~/ILSVRC2012/ --dataset ImageNet --depth 18 --save ./logs/diracnet_$RANDOM$RANDOM \
                --batchSize 256 --epoch_step [30,60,90] --epochs 100 --weightDecay 0.0001 --lr_decay_ratio 0.1
```


### nn.Module code

We provide `DiracConv1d`, `DiracConv2d`, `DiracConv3d`, which work like `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, but have Dirac-parametrization inside (our training code doesn't use these modules though).


### Pretrained models

We fold batch normalization and Dirac parameterization into `F.conv2d` `weight` and `bias` tensors for simplicity. Resulting models are as simple as VGG or AlexNet, having only nonlinearity+conv2d as a basic block.

See [diracnets.ipynb](diracnets.ipynb) for functional and modular model definitions.

There is also folded DiracNet definition in `diracnet.py`, which uses code from PyTorch model_zoo and downloads pretrained model from Amazon S3:

```python
from diracnet import diracnet18
model = diracnet18(pretrained=True)
```

Printout of the model above:

```
DiracNet(
  (features): Sequential(
    (conv): Conv2d (3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (max_pool0): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
    (group0.block0.relu): ReLU()
    (group0.block0.conv): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group0.block1.relu): ReLU()
    (group0.block1.conv): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group0.block2.relu): ReLU()
    (group0.block2.conv): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group0.block3.relu): ReLU()
    (group0.block3.conv): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (max_pool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (group1.block0.relu): ReLU()
    (group1.block0.conv): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group1.block1.relu): ReLU()
    (group1.block1.conv): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group1.block2.relu): ReLU()
    (group1.block2.conv): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group1.block3.relu): ReLU()
    (group1.block3.conv): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (max_pool2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (group2.block0.relu): ReLU()
    (group2.block0.conv): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group2.block1.relu): ReLU()
    (group2.block1.conv): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group2.block2.relu): ReLU()
    (group2.block2.conv): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group2.block3.relu): ReLU()
    (group2.block3.conv): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (max_pool3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (group3.block0.relu): ReLU()
    (group3.block0.conv): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group3.block1.relu): ReLU()
    (group3.block1.conv): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group3.block2.relu): ReLU()
    (group3.block2.conv): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (group3.block3.relu): ReLU()
    (group3.block3.conv): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (last_relu): ReLU()
    (avg_pool): AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True)
  )
  (fc): Linear(in_features=512, out_features=1000)
)
```

The models were trained with OpenCV, so you need to use it too to reproduce stated accuracy.

Pretrained weights for DiracNet-18 and DiracNet-34:<br>
<https://s3.amazonaws.com/modelzoo-networks/diracnet18v2folded-a2174e15.pth><br>
<https://s3.amazonaws.com/modelzoo-networks/diracnet34v2folded-dfb15d34.pth>

Pretrained weights for the original (not folded) model,  functional definition only:<br>
<https://s3.amazonaws.com/modelzoo-networks/diracnet18-v2_checkpoint.pth><br>
<https://s3.amazonaws.com/modelzoo-networks/diracnet34-v2_checkpoint.pth>

We plan to add more pretrained models later.

## Bibtex

```
@inproceedings{Zagoruyko2017diracnets,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {DiracNets: Training Very Deep Neural Networks Without Skip-Connections},
    url = {https://arxiv.org/abs/1706.00388},
    year = {2017}}
```
