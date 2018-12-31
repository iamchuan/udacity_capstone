# udacity_capstone

## Libraries:
- torch
- torchvision
- imageio
- matplotlib

## Datasets
- MNIST: `torchvision.datasets.MNIST`
- FashionMNIST: `torchvision.datasets.FashionMNIST`
- CartoonSet: [https://google.github.io/cartoonset/download.html]()

## Sample output

| Model |*Vanilla GAN*|*DCGAN*|*Conditional GAN*|*Conditional DCGAN*|
|---------|-------------|-------|-----------------|-------------------|
| MNIST |![](https://github.com/iamchuan/udacity_capstone/blob/master/images/mnist/vanillagan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/mnist/dcgan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/mnist/cgan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/mnist/cdcgan/animation.gif)|
| FID scores| 29.8860039314948 | 7.548617460620278 | 27.60618076139116 | 7.420052673632085 |
| FashionMNIST |![](https://github.com/iamchuan/udacity_capstone/blob/master/images/fashionmnist/vanillagan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/fashionmnist/dcgan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/fashionmnist/cgan/animation.gif)|![](https://github.com/iamchuan/udacity_capstone/blob/master/images/fashionmnist/cdcgan/animation.gif)|
| FID scores | 55.893088118384526 | 13.996017101103291 | 60.76478200199051 | 13.584223451920764 |
