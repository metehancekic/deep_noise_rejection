

import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deepillusion.torchattacks import PGD

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}, \usepackage{sfmath}')


class FeatureModel(nn.Module):
    def __init__(self, net):
        super(FeatureModel, self).__init__()
        self.first_layer = list(net.children())[0]  # firstlayer
        self.second_layer = list(net.children())[1]  # secondlayer

    def forward(self, image):
        x = F.max_pool2d(F.relu(self.first_layer(image)), (2, 2))
        x = F.relu(self.second_layer(x))
        return x


def plot_perturbations(args, net, x, y, x_adv, norm, budget, x_min, x_max, title=None):

    y_clean = net(x)
    y_adv = net(x_adv)

    pred_clean = y_clean.argmax(dim=1, keepdim=True)
    pred_adv = y_adv.argmax(dim=1, keepdim=True)

    x = (x+x_min)*(x_max - x_min)
    x_adv = (x_adv+x_min)*(x_max - x_min)

    x = x.view(-1, x.shape[-1], x.shape[-1])
    x_adv = x_adv.view(-1, x.shape[-1], x.shape[-1])

    fig = plt.figure(figsize=(10, 10))

    if norm == 'infty':
        plt.suptitle(r"Adversarial Attack with PGD($l_{\infty}$)", y=0.995)
    elif norm == 2:
        plt.suptitle(r"Adversarial Attack with PGD($l_{2}$)", y=0.995)
    if title:
        plt.title(title)

    for i in range(5):

        plt.subplot(5, 3, i*3+1)
        plt.imshow(x[i].cpu().numpy(), vmin=0, vmax=1)
        plt.title(f'Original Image {pred_clean[i].cpu().detach().numpy()}')
        plt.axis("off")

        plt.subplot(5, 3, i*3+2)
        plt.imshow(x_adv[i].cpu().numpy(), vmin=0, vmax=1)
        plt.title(f'Perturbed Image {pred_adv[i].cpu().detach().numpy()}')
        plt.axis("off")

        plt.subplot(5, 3, i*3+3)
        plt.imshow(x_adv[i].cpu().numpy()-x[i].cpu().numpy())
        plt.title(r"Perturbation $l_{:}$ = {:.0f}".format(norm, budget))
        plt.colorbar()
        plt.axis("off")

        plt.savefig(args.directory + 'figures/' + 'l_{:}_attack_new'.format(norm))


def plot_filters(args, net, fig_name=''):

    # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow(net.conv1.weight.view(-1, net.conv1.weight.shape[-1], net.conv1.weight.shape[-1]).cpu(
            ).detach().numpy()[i], cmap='viridis', vmin=net.conv1.weight.min(), vmax=net.conv1.weight.max())
        ax.set_title(f'Filter {i}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig(args.directory + 'figures/' + fig_name + '_layer1')

    ###############################################################################
    #---------------------------------- Layer 2 ----------------------------------#
    ###############################################################################

    # fig, axes = plt.subplots(nrows=10, ncols=20, figsize=(40, 20))
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(20, 10))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow(net.conv2.weight.view(-1, net.conv2.weight.shape[-1], net.conv2.weight.shape[-1]).cpu(
            ).detach().numpy()[i], cmap='viridis', vmin=net.conv2.weight.min(), vmax=net.conv2.weight.max())
        ax.set_title(f'Filter {i}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig(args.directory + 'figures/' + fig_name + '_layer2')


def plot_features(args, net, test_loader, device, data_params, attack_params, fig_name):

    fmodel = FeatureModel(net)

    fmodel.eval()
    images, labels = next(iter(test_loader))
    img = images[:1]
    lbl = labels[:1]
    img, lbl = img.to(device), lbl.to(device)

    img = img.view(1, img.shape[1], img.shape[2], img.shape[3])

    output = fmodel(img)
    out = net(img)
    pred = out.argmax(dim=1, keepdim=True)
    print(f'CLEAN, Prediction: {pred.cpu().detach().numpy()}, Original Label: {lbl.cpu().detach().numpy()}')

    # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
    # fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))

    fig, axes = plt.subplots(nrows=10, ncols=20, figsize=(40, 20))

    clean_features = output.view(-1, output.shape[-1], output.shape[-1]).cpu().detach().numpy()
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow(clean_features[i], cmap='viridis', vmin=output.min(), vmax=output.max())
        ax.set_title(f'Feature {i}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig(args.directory + 'figures/' + fig_name + '_features_layer2')

    ######## Adversarial Image ############

    pgd_args = dict(net=net,
                    x=img,
                    y_true=lbl,
                    verbose=False,
                    data_params=data_params,
                    attack_params=attack_params)
    perturbs = PGD(**pgd_args)

    img += perturbs
    img = torch.clamp(img, 0, 1)

    output = fmodel(img)
    adversarial_features = output.view(-1,
                                       output.shape[-1], output.shape[-1]).cpu().detach().numpy()
    out = net(img)
    pred = out.argmax(dim=1, keepdim=True)
    print(f'ADV, Prediction: {pred.cpu().detach().numpy()}, Original Label: {lbl.cpu().detach().numpy()}')

    # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
    # fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))

    fig, axes = plt.subplots(nrows=10, ncols=20, figsize=(40, 20))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow(adversarial_features[i], cmap='viridis',
                       vmin=output.min(), vmax=output.max())
        ax.set_title(f'Feature {i}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig(args.directory + 'figures/' + fig_name + '_features_layer2_adv_img')

    fig, axes = plt.subplots(nrows=10, ncols=20, figsize=(40, 20))

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        im = ax.imshow((normalize(adversarial_features) - normalize(clean_features))[i], cmap='viridis', vmin=(normalize(
            adversarial_features) - normalize(clean_features)).min(), vmax=(normalize(adversarial_features) - normalize(clean_features)).max())
        ax.set_title(f'Feature {i}')
        plt.axis("off")

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.1, hspace=0.25)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    plt.savefig(args.directory + 'figures/' + fig_name + '_features_layer2_adv_diff_img')

    small_value = 0.00001
    print(np.sum(np.abs(normalize(adversarial_features) - normalize(clean_features)),
                 axis=(1, 2))/(np.sum(normalize(clean_features), axis=(1, 2)) + small_value))


def normalize(x):
    small_value = 0.00001
    return x/(np.sum(np.abs(x), axis=(1, 2))+small_value).reshape(-1, 1, 1)
