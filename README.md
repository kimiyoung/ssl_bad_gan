
# Good Semi-Supervised Learning that Requires a Bad GAN

This is the code we used in our paper

>[Good Semi-supervised Learning that Requires a Bad GAN](https://arxiv.org/abs/1705.09783)

>Zihang Dai*, Zhilin Yang*, Fan Yang, William W. Cohen, Ruslan Salakhutdinov (*: equal contribution)

>NIPS 2017

## Requirements

The repo supports python 2.7 + pytorch 0.1.12. To install pytorch 0.1.12, run `conda install pytorch=0.1.12 cuda80 -c soumith`.

## Get Pretrained PixelCNN Model

```
mkdir model
cd model
wget http://kimi.ml.cmu.edu/mnist.True.3.best.pixel
```

## Run the Code

To reproduce our results on MNIST
```
python mnist_trainer.py
```

To reproduce our results on SVHN
```
python svhn_trainer.py
```

To reproduce our results on CIFAR-10
```
python cifar_trainer.py
```

## Results

Here is a comparison of different models using standard architectures without ensembles (100 labels on MNIST, 1000 labels on SVHN, and 4000 labels on CIFAR):

Method | MNIST (# errors) | SVHN (% errors) | CIFAR (% errors)
-- | -- | -- | --
CatGAN | 191 +/- 10 | - | 19.58 +/- 0.46
SDGM | 132 +/- 7 | 16.61 +/- 0.24 | -
Ladder Network | 106 +/- 37 | - | 20.40 +/- 0.47
ADGM | 96 +/- 2 | 22.86 | -
FM | 93 +/- 6.5 | 8.11 +/- 1.3 | 18.63 +/- 2.32
ALI | - | 7.42 +/- 0.65 | 17.99 +/- 1.62
VAT small | 136 | 6.83 | 14.87
Ours | **79.5 +/- 9.8** | **4.25 +/- 0.03** | **14.41 +/- 0.30**

