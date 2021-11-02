from __future__ import print_function
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import antialiased_cnns


def default_loader(path):
    return Image.open(path).convert("RGB")


class MyDataset(Dataset):
    def __init__(
        self,
        txt,
        transform=None,
        target_transform=None,
        loader=default_loader,
        image_folder="training_images/",
    ):
        fh = open(txt, "r")
        imgs = []
        for line in fh:
            line = line.strip("\n")
            line = line.rstrip()
            words = line.split()
            if image_folder == "training_images/":
                imgs.append((image_folder + words[0], int(words[1][:3]) - 1))
            else:
                imgs.append((image_folder + words[0], 0))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    trigger_times,
    last_loss,
    patience,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        if loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                return trigger_times, last_loss
        else:
            trigger_times = 0

        last_loss = loss

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
    return trigger_times, last_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="N",
        help="input patience for training (default: 10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # truth of training
    truth_of_training = []
    fh = open("training_labels.txt", "r")
    for line in fh:

        truth_of_training.append(int(line[9:12]) - 1)
    fh.close()

    model = antialiased_cnns.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, 200)

    # freeze the layers
    child_counter = 0
    for child in model.children():
        if child_counter < 6:
            print("child ", child_counter, " was frozen")
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == 6:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 3:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
                    print(
                        "child ",
                        children_of_child_counter,
                        "of child",
                        child_counter,
                        " was frozen",
                    )
                else:
                    print(
                        "child ",
                        children_of_child_counter,
                        "of child",
                        child_counter,
                        " was not frozen",
                    )
                children_of_child_counter += 1

        else:
            print("child ", child_counter, " was not frozen")
        child_counter += 1

    model.to(device)

    # augmentation
    transform_aug = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
                ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((300, 300)),
            transforms.RandomAffine(
                degrees=(-60, 60),
                translate=(0.15, 0.15),
                scale=(0.3, 1.8),
                shear=(0.2)
            ),
            transforms.CenterCrop(300),
        ]
    )

    train_data = MyDataset(txt="training_labels.txt", transform=transform_aug)
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False
    )

    trigger_times = 0
    last_loss = 10

    # train
    for epoch in range(1, args.epochs + 1):
        trigger_times, last_loss = train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            trigger_times,
            last_loss,
            args.patience,
        )
        scheduler.step()
        
    torch.save(model, "model.pt")


if __name__ == "__main__":
    main()
