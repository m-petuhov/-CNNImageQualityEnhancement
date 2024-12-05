import os
import cv2
import numpy as np

# test folders for raw and compressed in yuv and png formats
testfolderRawYuv = './testrawyuv/'
testfolderRawPng = './testrawpng/'
testfolderCompYuv = './testcompyuv/'
testfolderCompPng = './testcomppng/'

# train folders for raw and compressed in yuv and png formats
trainfolderRawYuv = './trainrawyuv/'
trainfolderRawPng = './trainrawpng/'
trainfolderCompYuv = './traincompyuv/'
trainfolderCompPng = './traincomppng/'


def yuv2rgb (Y,U,V,fw,fh):
    U_new = cv2.resize(U, (fw, fh),cv2.INTER_CUBIC)
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


def FromFolderYuvToFolderPNG(folderyuv, folderpng, fw, fh):
    dir_list = os.listdir(folderpng)
    for name in dir_list:
        os.remove(folderpng + name)
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    Im = np.zeros((fh, fw, 3))
    dir_list = os.listdir(folderyuv)
    pngframenum = 0
    for name in dir_list:
        fullname = folderyuv + name
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
            print(fullname, frames)
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
                r, g, b = yuv2rgb(Y, U, V, fw, fh)

                Im[:, :, 0] = b
                Im[:, :, 1] = g
                Im[:, :, 2] = r
                pngfilename = "%s/%i.png" % (folderpng, pngframenum)
                cv2.imwrite(pngfilename, Im)
                pngframenum = pngframenum + 1
            fp.close()
    return (pngframenum - 1)


if __name__ == '__main__':
    for d in [trainfolderRawPng, trainfolderCompPng, testfolderRawPng, testfolderCompPng]:
        os.makedirs(d, exist_ok=True)

    w = 480
    h = 320
    FromFolderYuvToFolderPNG(trainfolderRawYuv, trainfolderRawPng, w, h)
    FromFolderYuvToFolderPNG(trainfolderCompYuv, trainfolderCompPng, w, h)
    FromFolderYuvToFolderPNG(testfolderRawYuv, testfolderRawPng, w, h)
    FromFolderYuvToFolderPNG(testfolderCompYuv, testfolderCompPng, w, h)