import torch
import utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from attack import *
from models import PytorchModel
from paper_models import MNIST, CIFAR10, VGG16
import torchvision.models as models
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import argparse
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Dataset to be tested, [MNIST, CIFAR10, ImageNet]')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack.')
parser.add_argument('--norm', type=str,  default='lf', help='Which lp constraint to run bandits. Options: [l2, lf]')
parser.add_argument('--eps', type=float, default=0.5, help='Distortion threshold')

parser.add_argument('--test_batch', type=int, default=1000,
                    help='The number of to-be-attacked images')
parser.add_argument('--attack_model', type=str, default='MNIST-CNN', help='paper_model to be attacked. '
                                                                          'Options: [MNIST-CNN, CIFAR10-CNN, CIFAR10-VGG16, ImageNet-ResNet50, ImageNet-VGG16]')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--stand', action='store_true', default=False, help='Standardization')
parser.add_argument('--query', type=int, default=20000, help='Query limit allowed')
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
args = parser.parse_args()

param_run = '{}_targeted_{}_norm_{}_eps_{}_test_number_{}_{}_seed_{}_nqueries_{}_gpu_{}'.format(
            args.dataset, args.targeted, args.norm, args.eps, args.test_batch,
            args.attack_model,args.seed, args.query, args.gpu)

#### env
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.set_device(args.gpu)
print('gpu:', args.gpu)


#### macros
attack_list = {
    "l2": Sign_OPT_Plus_l2,
    "lf": Sign_OPT_Plus_lf,
}

l2_list = ["Sign_OPT_Plus"]
linf_list = ["Sign_OPT_Plus_lf"]

#### load dataset
if args.dataset == "MNIST":
    train_dataset = dataset.MNIST(root='./dataset/MNIST', download=True, train=True,
                                  transform=transforms.ToTensor())
    test_dataset = dataset.MNIST(root='./dataset/MNIST', download=True, train=False,
                                 transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    num_classes = 10
elif args.dataset == 'CIFAR10':
    train_dataset = dataset.CIFAR10(root='./dataset/CIFAR10', download=True, train=True,
                                   transform=transforms.ToTensor())
    test_dataset = dataset.CIFAR10(root='./dataset/CIFAR10', download=True, train=False,
                                   transform= transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    num_classes = 10
elif args.dataset == 'ImageNet':
    # TODO: change the below to point to the ImageNet validation set,
    # formatted for PyTorch ImageFolder
    # Instructions for how to do this can be found at:
    # https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
    IMAGENET_PATH = '/path/to/ImageNet/validation/set/'
    test_dataset = ImageFolder(IMAGENET_PATH, utils.IMAGENET_TRANSFORM)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    num_classes = 1000
else:
    print("Unsupport dataset")

#### load paper_model
if args.attack_model == 'MNIST-CNN':
    model = MNIST()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.load_state_dict(torch.load('./paper_model/mnist.pt', map_location={'cuda:0': 'cuda:'+str(args.gpu)}))
elif args.attack_model == 'CIFAR10-CNN':
    model = CIFAR10()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.load_state_dict(torch.load('./paper_model/cifar10.pt', map_location={'cuda:0': 'cuda:'+str(args.gpu)}))
elif args.attack_model == 'CIFAR10-VGG16':
    model = VGG16()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
    model.load_state_dict(torch.load('./paper_model/cifar10_vgg.pt', map_location={'cuda:0': 'cuda:'+str(args.gpu)}))
elif args.attack_model == 'ImageNet-ResNet50':
    model = models.resnet50(pretrained=True).cuda()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])
elif args.attack_model == 'ImageNet-VGG16':
    model = models.vgg16(pretrained=True).cuda()
    model = torch.nn.DataParallel(model, device_ids=[args.gpu])

amodel = PytorchModel(model, bounds=[0, 1], num_classes=num_classes, dataset=args.dataset, stand=args.stand)
model.cuda()
model.eval()

logsdir = './logs'
if not args.targeted:
    attack = attack_list[args.norm](amodel, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7],
                                                                                   param_run), targ_flag=args.targeted)
else:
    if args.dataset == "MNIST" or args.dataset == "CIFAR10":
        initial_dataset = train_dataset
    elif args.dataset == "ImageNet":
        initial_dataset = test_dataset
    attack = attack_list[args.norm](amodel, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7],
                                                                                   param_run), initial_dataset=initial_dataset, targ_flag=args.targeted)

suc_count, fail_count, unclean_count = 0, 0, 0
dis_sum = 0

for i, (images, labels) in enumerate(test_loader):
    # logging.info(f"image batch: {i}")
    attack.logger.log("image batch:{}".format(i))
    ## dataset
    if i == args.test_batch: break
    xi, yi = images.cuda(), labels.cuda()
    ## attack
    theta_init = None
    x_adv, dis, suc, fail, unclean = attack(xi, yi, distortion=args.eps, query_limit=args.query)
    x_adv = x_adv.clamp(0,1)

    suc_count += suc
    fail_count += fail
    unclean_count += unclean
    SR = suc_count / (suc_count + fail_count + 1e-8)
    attack.logger.log("suc:{}, fail:{}, unclean:{}, SR:{:.4f}".format(suc_count, fail_count, unclean_count, SR))
    if(unclean != 1):
        dis_sum += dis
        dis_avg = dis_sum/(suc_count + fail_count)
        attack.logger.log("Avgl2:{:.4f}".format(dis_avg))

attack.logger.log('='*80)
attack.logger.log("SR:{:.4f}, Avgl2:{:.4f}".format(SR, dis_avg))

