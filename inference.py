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


class MyDataset(Dataset):
    def __init__(
        self,
        txt,
        transform=None,
        target_transform=None,
        loader=None,
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
        self.loader = self.default_loader

    def default_loader(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def make_answer(model, device,filename):


    fh = open('classes.txt', 'r')
    number_to_bird = {}

    n = 0
    for line in fh:
        line = line.strip('\n')
        number_to_bird[int(line[0:3])] = line

    fh.close()

    transform_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Resize((300,300)),
        # transforms.ToPILImage()

        
        ]
    )
    test_data=MyDataset(txt='testing_img_order.txt', transform=transform_ori, image_folder = 'testing_images/')
    test_loader = DataLoader(test_data, 1,shuffle=False,num_workers = 8,pin_memory=True)
    model.to(device)
    model.eval()

    ans = []
    picture_name = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            ans.append(pred[0][0])  # get the index of the max log-probability
            
    fh = open('testing_img_order.txt', 'r')
    for line in fh:
        line = line.strip('\n')
        picture_name.append(line)

    fh.close()
    
    fh = open(filename, 'w')
    for i in range(len(ans)):
        line = str(picture_name[i]) + ' '+ str(number_to_bird[ans[i].item()+1])+'\n'
        fh.write(line)
    fh.close()

def main():
    device = torch.device("cuda")
    torch.manual_seed(21)
    model = torch.load('model.pt')
    make_answer(model,device,'answer.txt')

if __name__ == '__main__':   
    main()
