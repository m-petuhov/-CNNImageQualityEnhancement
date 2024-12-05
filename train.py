import torch
from model import UNet

import torchvision.transforms.functional as TF
import torchvision.transforms.v2.functional as TFV2
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import cv2
import PIL
import numpy as np
import torch.nn as nn
import torch.optim
import random
import os
from PIL import Image
from numpy import *


def LoadImagesFromFolder(foldername, dir_list):
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        Nmax = Nmax + 1

    x = np.zeros([Nmax, 320, 480, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        I1 = cv2.imread(fullname)
        x[N, :, :, 0] = I1[:, :, 2]
        x[N, :, :, 1] = I1[:, :, 1]
        x[N, :, :, 2] = I1[:, :, 0]
        N = N + 1
    return x


class MyDataset(Dataset):
    def __init__(self, Xraw, Xcomp):
        self.xraw = Xraw
        self.xcomp = Xcomp

    def transform(self, xraw, xcomp):
        xraw = Image.fromarray((xraw * 255).astype(np.uint8))
        xcomp = Image.fromarray((xcomp * 255).astype(np.uint8))

        # Random horizontal flipping
        if random.random() > 0.5:
            xraw = TF.hflip(xraw)
            xcomp = TF.hflip(xcomp)

        # Random vertical flipping
        if random.random() > 0.5:
            xraw = TF.vflip(xraw)
            xcomp = TF.vflip(xcomp)

        # Randomm affine transform
        if random.random() > 0.5:
            angle = random.randint(0, 360)
            translate = (random.randint(100, 300) / 1000, random.randint(100, 300) / 1000)
            scale = random.randint(500, 700) / 1000

            xraw = TFV2.affine(xraw, angle=angle, translate=translate, scale=scale, shear=(0.0, 0.0))
            xcomp = TFV2.affine(xcomp, angle=angle, translate=translate, scale=scale, shear=(0.0, 0.0))

        # Randomm jitter
        if random.random() > 0.5:
            jitter = v2.ColorJitter(brightness=.5, hue=.3)

            brightness = jitter._check_input(.5, "brightness")
            contrast = jitter._check_input(0, "contrast")
            saturation = jitter._check_input(0, "saturation")
            hue = jitter._check_input(.3, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = v2.ColorJitter.get_params(
                brightness, contrast, saturation, hue
            )

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    xraw = TFV2.adjust_brightness(xraw, brightness_factor)
                    xcomp = TFV2.adjust_brightness(xcomp, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    xraw = TFV2.adjust_contrast(xraw, contrast_factor)
                    xcomp = TFV2.adjust_brightness(xcomp, brightness_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    xraw = TFV2.adjust_saturation(xraw, saturation_factor)
                    xcomp = TFV2.adjust_saturation(xcomp, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    xraw = TFV2.adjust_hue(xraw, hue_factor)
                    xcomp = TFV2.adjust_hue(xcomp, hue_factor)

        # Transform to tensor
        xraw = TF.to_tensor(xraw)
        xcomp = TF.to_tensor(xcomp)
        return xraw, xcomp

    def __getitem__(self, index):
        x, y = self.transform(self.xraw[index], self.xcomp[index])
        return x, y

    def __len__(self):
        return len(self.xraw)


if __name__ == '__main__':
    model = UNet(3, 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    loss_fn = torch.nn.MSELoss()
    model.to(device)
    best_loss = 100

    print('Loading raw train images...')
    train_dir_list = os.listdir('./trainrawpng/')
    random.shuffle(train_dir_list)
    Xraw = LoadImagesFromFolder('./trainrawpng/', train_dir_list)
    print('Loading compressed train images...')
    Xcomp = LoadImagesFromFolder('./traincomppng/', train_dir_list)
    Xraw = Xraw / 255.0
    Xcomp = Xcomp / 255.0

    print('Loading raw validiation images...')
    val_dir_list = os.listdir('./testrawpng/')
    XrawVal = LoadImagesFromFolder('./testrawpng/', val_dir_list)
    print('Loading compressed validiation images...')
    XcompVal = LoadImagesFromFolder('./testcomppng/', val_dir_list)
    XrawVal = XrawVal / 255.0
    XcompVal = XcompVal / 255.0

    train_dataset = MyDataset(Xraw, Xcomp)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for optimizer in [torch.optim.Adam(model.parameters(), lr=1e-3), torch.optim.Adam(model.parameters(), lr=1e-4)]:
        for _ in range(50):
            # Train
            running_loss = 0.0
            model.train()

            for xraw, xcomp in train_dataloader:
                batch = xcomp.to(device, torch.float)
                labels = xraw.to(device, torch.float)

                optimizer.zero_grad()

                predictions = model(batch)
                loss = loss_fn(predictions, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)

            print("Train loss", running_loss / len(Xraw))

            # Eval
            running_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i in range(0, len(XrawVal), batch_size):
                    batch = torch.from_numpy(XcompVal[i:i + batch_size]).permute(0, 3, 1, 2).to(device, torch.float)
                    labels = torch.from_numpy(XrawVal[i:i + batch_size]).permute(0, 3, 1, 2).to(device, torch.float)

                    predictions = model(batch)
                    loss = loss_fn(predictions, labels)

                    running_loss += loss.item() * batch.size(0)

            print("Val loss", running_loss / len(XrawVal))
            if best_loss > running_loss / len(XrawVal):
                torch.save(model.state_dict(), 'best_model.pth')
                best_loss = running_loss / len(XrawVal)
