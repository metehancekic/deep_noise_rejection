"""

Example Run

python -m deep_noise_rejection.CIFAR10.main --model ResNetMadry -tra RFGSM -at -Ni 7 -tr -sm

"""
import time
import os
from os import path
from tqdm import tqdm
import numpy as np

import logging

from apex import amp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD

# CIFAR10 TRAIN TEST CODES
from deep_noise_rejection.CIFAR10.models.resnet import ResNet34
from deep_noise_rejection.CIFAR10.models.resnet_new import ResNet, ResNetWide
from deep_noise_rejection.CIFAR10.models.preact_resnet import PreActResNet18
from deep_noise_rejection.train_test_functions import train, test

from deep_noise_rejection.CIFAR10.parameters import get_arguments
from deep_noise_rejection.CIFAR10.read_datasets import cifar10


logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

    args = get_arguments()

    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.directory + 'logs/' + args.model +
                                '_' + args.tr_attack + '.log'),
            logging.StreamHandler()
            ])
    logger.info(args)
    logger.info("\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = cifar10(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    if args.model == "ResNet":
        model = ResNet34().to(device)
    elif args.model == "ResNetMadry":
        model = ResNet().to(device)
    elif args.model == "ResNetMadryWide":
        model = ResNetWide().to(device)
    elif args.model == "ResNet18":
        model = PreActResNet18().to(device)
    else:
        raise NotImplementedError

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # logger.info(model)
    # logger.info("\n")

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)
    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM)

    attack_params = {
        "norm": args.tr_norm,
        "eps": args.tr_epsilon,
        "alpha": args.tr_alpha,
        "step_size": args.tr_step_size,
        "num_steps": args.tr_num_iterations,
        "random_start": args.tr_rand,
        "num_restarts": args.tr_num_restarts,
        }

    data_params = {"x_min": 0., "x_max": 1.}

    adversarial_args = dict(attack=attacks[args.tr_attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             verbose=False))

    # Checkpoint Namer
    checkpoint_name = args.model
    if adversarial_args["attack"]:
        for key in attack_params:
            checkpoint_name += "_" + str(key) + "_" + str(attack_params[key])
    checkpoint_name += ".pt"

    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
    if args.train:
        logger.info(args.tr_attack + " training")
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_args = dict(model=model,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              adversarial_args=adversarial_args)
            train_loss, train_acc = train(**train_args)

            test_args = dict(model=model,
                             test_loader=test_loader)
            test_loss, test_acc = test(**test_args)

            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        # Save model parameters
        if args.save_model:
            if not os.path.exists(args.directory + "checkpoints/"):
                os.makedirs(args.directory + "checkpoints/")
            torch.save(model.state_dict(), args.directory + "checkpoints/" + checkpoint_name)

    else:
        model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))

        print("Clean test accuracy")
        test_args = dict(model=model,
                         test_loader=test_loader)
        test_loss, test_acc = test(**test_args)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    if args.attack_network:
        attack_params = {
            "norm": args.norm,
            "eps": args.epsilon,
            "alpha": args.alpha,
            "step_size": args.step_size,
            "num_steps": args.num_iterations,
            "random_start": args.rand,
            "num_restarts": args.num_restarts,
            }

        adversarial_args = dict(attack=attacks[args.attack],
                                attack_args=dict(net=model,
                                                 data_params=data_params,
                                                 attack_params=attack_params,
                                                 verbose=True))

        for key in attack_params:
            logger.info(key + ': ' + str(attack_params[key]))

        test_args = dict(model=model,
                         test_loader=test_loader,
                         adversarial_args=adversarial_args,
                         verbose=True)
        test_loss, test_acc = test(**test_args)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')

    # if args.black_box:
    #     attack_loader = cifar10_black_box(args)

    #     test(model, attack_loader)


if __name__ == "__main__":
    main()
