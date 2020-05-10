"""
Authors: Metehan Cekic
Date: 2020-03-09

Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm
from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, train_loader, optimizer, scheduler, adversarial_args=None):

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def test(model, test_loader, adversarial_args=None, verbose=False):

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if verbose:
        iter_test_loader = tqdm(test_loader)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs
        # breakpoint()

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size
