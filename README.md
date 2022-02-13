# Detecting Corrupted Labels Without Training a Model to Predict

This code is a PyTorch implementation of the [paper](https://arxiv.org/abs/2110.06283): Detecting Corrupted Labels Without Training a Model to Predict 


## Prerequisites

Python 3.6.9

PyTorch 1.7.1

Torchvision 0.8.2

Full list in ./requirements.txt

Datasets will be downloaded to ./data/.

## Run HOC + Vote Based and Rank Based method

On CIFAR-10 .

```
sh ./test_c10_instance.sh  
```

On CIFAR-100

```
sh ./test_c100_instance.sh  
```
