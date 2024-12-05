import torch
import os
import numpy as np
import cv2
from model import UNet
from tqdm import tqdm


def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def yuv2rgb(Y, U, V, fw, fh):
    U_new = cv2.resize(U, (fw, fh), cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), cv2.INTER_CUBIC)
    U = U_new
    V = V_new
    Y = Y
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    for m in range(fh):
        for n in range(fw):
            if (rf[m, n] > 255):
                rf[m, n] = 255
            if (gf[m, n] > 255):
                gf[m, n] = 255
            if (bf[m, n] > 255):
                bf[m, n] = 255
            if (rf[m, n] < 0):
                rf[m, n] = 0
            if (gf[m, n] < 0):
                gf[m, n] = 0
            if (bf[m, n] < 0):
                bf[m, n] = 0
    r = rf
    g = gf
    b = bf
    return r, g, b


def GetRGBFrame(folderyuv, VideoNumber, FrameNumber, fw, fh):
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = folderyuv + name
        if v != VideoNumber:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                if f == FrameNumber:
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)
                    return r, g, b


def ShowFramePSNRPerformance(folderyuv, foldercomp, VideoIndex, framesmax, fw, fh, device):
    RGBRAW = np.zeros((fh, fw, 3))
    RGBCOMP = np.zeros((fh, fw, 3))
    dir_list = os.listdir(folderyuv)
    v = 0
    l = []
    for name in dir_list:
        fullname = folderyuv + name
        print(name)
        if v != VideoIndex:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            frames = (2 * size) // (fw * fh * 3)
            if frames > framesmax:
                frames = framesmax

            PSNRCOMP = np.zeros((frames))
            PSNRUNET = np.zeros((frames))
            for f in tqdm(range(frames)):
                r, g, b = GetRGBFrame(folderyuv, VideoIndex, f, fw, fh)
                RGBRAW[:, :, 0] = r
                RGBRAW[:, :, 1] = g
                RGBRAW[:, :, 2] = b
                r, g, b = GetRGBFrame(foldercomp, VideoIndex, f, fw, fh)
                RGBCOMP[:, :, 0] = r
                RGBCOMP[:, :, 1] = g
                RGBCOMP[:, :, 2] = b
                PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)

                out = unet(
                    torch.from_numpy(RGBCOMP / 255.0).unsqueeze(0).permute(0, 3, 1, 2).to(device, torch.float).to(
                        device, torch.float)).permute(0, 2, 3, 1)[0]
                out = out.cpu().detach().numpy()
                PSNRUNET[f] = cal_psnr(RGBRAW / 255.0, out)

        return PSNRCOMP, PSNRUNET


if __name__ == "__main__":
    unet = UNet(in_channels=3, out_channels=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet.load_state_dict(torch.load('unet_best.pth', weights_only=True))
    unet.eval()

    PSNRCOMP, PSNRUNET = ShowFramePSNRPerformance('./testrawyuv/', './testcompyuv/', 0, 100, 480, 320, device)
    print(f"Средний psnr сжатого изображения - {np.mean(PSNRCOMP):.2f}")
    print(f"Средний psnr улучшенного сжатого изображения - {np.mean(PSNRUNET):.2f}")