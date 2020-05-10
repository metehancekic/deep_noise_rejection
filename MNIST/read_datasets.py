import numpy as np
from os import path


import torch
import torch.nn as nn
from torchvision import datasets, transforms


def MNIST(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.directory + "data",
                train=True,
                download=False,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.directory + "data",
                train=False,
                download=False,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
            )

    elif args.dataset == "fashion":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.directory + "data",
                train=True,
                download=False,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.directory + "data",
                train=False,
                download=False,
                transform=transforms.Compose([transforms.ToTensor(), ]),
                ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
            )

    return train_loader, test_loader


def MNIST_demixed(args):

    # test_blackbox1 = np.load('/home/canbakiskan/cvx_demixing/data/MNIST/preprocessed_test_dataset/cvxpy_zero_rm_ps_5_st_1_a_1.0_n_200_eps_0_l_0.10.npz')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    test_blackbox = np.load(
        "/home/canbakiskan/cvx_demixing/data/MNIST/preprocessed_test_dataset/cvxpy_zero_rm_ps_4_st_2_a_1.0_n_200_eps_0_l_0.10.npz"
        )

    test_madry = np.load(
        "/home/canbakiskan/cvx_demixing/data/MNIST/attacked_dataset/NT_attack.npy"
        )

    test_clean = np.load(
        "/home/canbakiskan/cvx_demixing/data/MNIST/preprocessed_test_dataset/CLEAN_cvxpy_zero_rm_ps_4_st_2_a_1.0_n_200_eps_0_l_0.10.npz"
        )

    train_set = np.zeros((60000, 28, 28))
    for i in range(6):
        train_dict = np.load(
            "/home/canbakiskan/cvx_demixing/data/MNIST/preprocessed_train_dataset/TRAIN_"
            + str(i * 10000)
            + "_"
            + str((i + 1) * 10000 - 1)
            + "_cvxpy_zero_rm_ps_4_st_2_a_1.0_n_200_eps_0_l_0.10.npz"
            )
        train_set[i * 10000: (i + 1) * 10000] = train_dict["reconstruction"]

    test_set_clean = test_clean["reconstruction"].copy()
    test_set_bb = test_blackbox["reconstruction"].copy()

    data_dir = "/home/canbakiskan/cvx_demixing/data/MNIST"
    mnist = datasets.MNIST(
        path.join(data_dir, "original_dataset"),
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        )

    y_train = mnist.targets.numpy()

    mnist = datasets.MNIST(
        path.join(data_dir, "original_dataset"),
        train=False,
        transform=None,
        target_transform=None,
        download=False,
        )

    y_test = mnist.targets.numpy()

    tensor_x = torch.Tensor(train_set).view(-1, 1, 28, 28)
    tensor_y = torch.Tensor(y_train).long()

    tensor_data = torch.utils.data.TensorDataset(
        tensor_x, tensor_y
        )  # create your datset
    train_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.batch_size, shuffle=True, **kwargs
        )  # create your dataloader

    tensor_x = torch.Tensor(test_set_clean).view(-1, 1, 28, 28)
    tensor_y = torch.Tensor(y_test).long()

    tensor_data = torch.utils.data.TensorDataset(
        tensor_x, tensor_y
        )  # create your datset
    test_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=True, **kwargs
        )  # create your dataloader

    tensor_x = torch.Tensor(test_set_bb).view(-1, 1, 28, 28)

    tensor_data = torch.utils.data.TensorDataset(
        tensor_x, tensor_y
        )  # create your datset
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=True, **kwargs
        )  # create your dataloader

    tensor_x = torch.Tensor(test_madry).view(-1, 1, 28, 28)
    tensor_data = torch.utils.data.TensorDataset(
        tensor_x, tensor_y
        )  # create your datset
    madry_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=True, **kwargs
        )  # create your dataloader

    return train_loader, test_loader, attack_loader, madry_loader
