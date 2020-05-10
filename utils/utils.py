"""
Utilities
PyTorch

Example Run

python -m deep_adv.utils.utils

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import os
from tqdm import tqdm

# from deep_adv.adversary.norm_ball_attacks import ProjectedGradientDescent as PGD
from deepillusion.torchattacks import PGD


def save_perturbed_images(args, model, device, data_loader, data_params, attack_params):

    # Set phase to testing
    model.eval()

    test_loss = 0
    correct = 0
    all_images = []
    all_labels = []
    all_preds = []
    for data, target in tqdm(data_loader):

        data, target = data.to(device), target.to(device)

        # Attacks
        pgd_args = dict(net=model,
                        x=data,
                        y_true=target,
                        data_params=data_params,
                        attack_params=attack_params)
        perturbs = PGD(**pgd_args)
        data += perturbs

        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        all_images.append(data.detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())

    # Divide summed-up loss by the number of datapoints in dataset
    test_loss /= len(data_loader.dataset)

    # Print out Loss and Accuracy for test set
    print(
        f"\nAdversarial test set (l_{attack_params['norm']}): Average loss: {test_loss:.2f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.2f}%)\n"
        )

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    if not os.path.exists(args.directory + "data/attacked_images/"):
        os.makedirs(args.directory + "data/attacked_images/")
    np.savez_compressed(
        args.directory + "data/attacked_images/" + args.model,
        images=all_images,
        labels=all_labels,
        preds=all_preds,
        )


def main():

    from deep_adv.CIFAR10.read_datasets import cifar10
    from deep_adv.CIFAR10.parameters import get_arguments
    from deep_adv.CIFAR10.models.resnet import ResNet34

    args = get_arguments()

    # Get same results for each training with same parameters !!
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = cifar10(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    if args.model == "ResNet":
        model = ResNet34().to(device)
    else:
        raise NotImplementedError

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.load_state_dict(
        torch.load(args.directory + "checkpoints/" + args.model + ".pt")
        )

    data_params = {"x_min": x_min, "x_max": x_max}
    attack_params = {
        "norm": "inf",
        "eps": args.epsilon,
        "step_size": args.step_size,
        "num_steps": args.num_iterations,
        "random_start": args.rand,
        "num_restarts": args.num_restarts,
        }

    save_perturbed_images(
        args,
        model,
        device,
        test_loader,
        data_params=data_params,
        attack_params=attack_params,
        )


if __name__ == "__main__":
    main()
