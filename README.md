# Sign-OPT-Plus   
---
**SIGN-OPT+: AN IMPROVED SIGN OPTIMIZATION ADVERSARIAL ATTACK**   
*<a href="mailto:ranyu@e.gzhu.edu.cn">Yu Ran</a> and Yuan-Gen Wang<sup>\*</sup>*  

## Setup
### Requirements
* Python 3.6.13 or above    
* PyTorch 1.4.0 or above   
### Datasets   
We evaluate the proposed method on the MNIST, CIFAR10, and ImageNet datasets.    
In main.py, set the following variable:    

* `IMAGENET_PATH`: path to the ImageNet validation set.    
##  Using models and attacks from the paper
The following provides the arguments to run the attacks described in the paper.  
### l2 norm untargeted attack
```bash
python main.py --norm l2 --dataset MNIST --attack_model MNIST-CNN --query 4000 --eps 1.5 --stand --gpu 0
```
```bash
python main.py --norm l2 --dataset CIFAR10 --attack_model CIFAR10-CNN --query 4000 --eps 0.5 --gpu 0
```
```bash
python main.py --norm l2 --dataset ImageNet --attack_model ImageNet-ResNet50 --query 4000 --eps 3.0 --stand --gpu 0
```
Note that the code of the untargeted attack in this repo has image standardization when the image is predicted. To fairly compare with Sign-OPT, Sign-OPT should add the standardization operation when making predictions.
### l2 norm targeted attack
```bash
python main.py --norm l2 --targeted --dataset MNIST --attack_model MNIST-CNN --query 8000 --eps 1.5 --gpu 0
```
```bash
python main.py --norm l2 --targeted --dataset CIFAR10 --attack_model CIFAR10-CNN --query 8000 --eps 0.5 --gpu 0
```
```bash
python main.py --norm l2 --targeted --dataset ImageNet --attack_model ImageNet-ResNet50 --query 10000 --eps 3.0 --gpu 0
```
### lf norm untargeted attack
```bash
python main.py --norm lf --dataset MNIST --attack_model MNIST-CNN --query 15000 --eps 0.3 --gpu 0
```
```bash
python main.py --norm lf --dataset CIFAR10 --attack_model CIFAR10-CNN --query 4000 --eps 0.03 --gpu 0
```
```bash
python main.py --norm lf --dataset ImageNet --attack_model ImageNet-ResNet50 --query 4000 --eps 0.03 --stand --gpu 0
```
### lf norm targeted attack
```bash
python main.py --norm lf --targeted --dataset MNIST --attack_model MNIST-CNN --query 20000 --eps 0.3 --gpu 0
```
```bash
python main.py --norm lf --targeted --dataset CIFAR10 --attack_model CIFAR10-CNN --query 8000 --eps 0.03 --gpu 0
```
```bash
python main.py --norm lf --targeted --dataset ImageNet --attack_model ImageNet-ResNet50 --query 10000 --eps 0.3 --gpu 0
```
## License
This source code is made available for research purposes only.
## Acknowledgment
Our code is built upon [**Sign-OPT**](https://github.com/cmhcbb/attackbox).