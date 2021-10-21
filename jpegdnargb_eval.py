"""Jpeg DNA RGB evaluation script"""

from pathlib import Path
from dataclasses import make_dataclass
import math
import pickle
from skimage import io
import pandas as pd
from pandas import ExcelWriter
import jpegdna
from jpegdna.codecs import JPEGDNARGB
from jpegdna.transforms import RGBYCbCr
from IQA_pytorch import SSIM, MS_SSIM, VIF
import numpy as np
import torch

def stats(func):
    """Stats printing and exception handling decorator"""

    def inner(*args):
        try:
            code, decoded = func(*args)
        except ValueError as err:
            print(err)
        else:
            MSE2 = 0
            compression_rate = 24 * img.shape[0] * img.shape[1] / len(code)
            channel_names = ["Y", "Cb", "Cr"]
            PSNR_s = [0] * 3
            color_conv = RGBYCbCr()
            img_ycbcr = color_conv.forward(img)
            decoded_ycbcr = color_conv.forward(decoded)
            for k in range(3):
                diff = (img_ycbcr[:, :, k].astype(int)-decoded_ycbcr[:, :, k].astype(int))
                MSE = 0
                for i in range(len(diff)):
                    for j in range(len(diff[0])):
                        MSE += diff[i, j]**2
                MSE /= len(diff)
                MSE /= len(diff[0])
                MSE2 += MSE
                PSNR = 10 * math.log10((255*255)/MSE)
                print(f"Mean squared error {channel_names[k]}: {MSE}")
                print(f"PSNR {channel_names[k]}: {PSNR}")
                PSNR_s[k] = PSNR
            MSE2 /= 3
            PSNR = 10 * math.log10((255*255)/MSE2)
            
            #Call the functions of SSIM, MS-SSIM, VIF
            D_1 = SSIM()
            D_2 = MS_SSIM()
            D_3 = VIF()
            
            #To get 4-dimension torch tensors, (N, 3, H, W)
            torch_decoded = torch.FloatTensor(decoded_ycbcr.swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)
            torch_img = torch.FloatTensor(img_ycbcr.swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)
            
            #Calculate SSIM, MS-SSIM, VIF
            SSIM_r = D_1(torch_decoded , torch_img, as_loss=False)
            MS_SSIM_r = D_2(torch_decoded, torch_img)
            VIF_r = D_3(torch_decoded, torch_img)
            
            #Print out the results
            print(f"Mean squared error: {MSE2}")
            print(f"General PSNR: {PSNR}")
            print(f"SSIM: {SSIM_r}")
            print(f"MS_SSIM: {MS_SSIM_r}")
            print(f"VIF: {VIF_r}")
            print(f"Compression rate: {compression_rate} bits/nt")
            # plt.imshow(decoded)
            # plt.show()
            # io.imsave(str(compression_rate) + ".png", decoded)
            return compression_rate, PSNR, PSNR_s, SSIM_r, MS_SSIM_r, VIF_r
    return inner

def encode_decode(img, alpha):
    """Function for encoding and decoding"""
    choice = "default"
    # Coding
    codec = JPEGDNARGB(alpha, verbose=False, verbosity=3)
    if choice == "from_img":
        (code, res) = codec.full_encode(img, "from_img")
    elif choice == "from_file":
        with open(Path(jpegdna.__path__[0] + "/data/freqs_rgb_4_2_2.pkl"), "rb") as file:
            freqs = pickle.load(file)
        (code, res) = codec.full_encode(img, "from_file", freqs['freq_dc'], freqs['freq_ac'])
    elif choice == "default":
        (code, res) = codec.full_encode(img, "default")
    # Decoding
    codec2 = JPEGDNARGB(alpha, verbose=False, verbosity=3)
    if choice == "from_img":
        params = (res[0][1:],
                  res[1][1:],
                  res[2][1:])
        decoded = codec2.full_decode(code, "from_img", params)
    elif choice == "from_file":
        with open(Path(jpegdna.__path__[0] + "/data/freqs_rgb_4_2_2.pkl"), "rb") as file:
            freqs = pickle.load(file)
        params = ((res[0][1], res[0][2], freqs['Y']['freq_dc'], freqs['Y']['freq_ac']),
                  (res[1][1], res[1][2], freqs['Cb']['freq_dc'], freqs['Cb']['freq_ac']),
                  (res[2][1], res[2][2], freqs['Cr']['freq_dc'], freqs['Cr']['freq_ac']))
        decoded = codec2.full_decode(code, "from_file", params)
    elif choice == "default":
        params = (res[0][1:3],
                  res[1][1:3],
                  res[2][1:3])
        decoded = codec2.full_decode(code, "default", params)
    return code, decoded

@stats
def experiment(img, alpha):
    """Full experiment with stats and exception handling"""
    return encode_decode(img, alpha)

if __name__ == '__main__':
    value = make_dataclass("value", [("Compressionrate", float), ("PSNR", float), ("PSNR_Y", float), ("PSNR_Cb", float), ("PSNR_Cr", float)
                           ,("SSIM_r", float), ("MS_SSIM_r", float), ("VIF_r", float)])
    general_results = []
    img_names = ["kodim01.png"]
    #for i in range(1, 25):
        #img_names.append(f"kodim{i:02d}.png")
    for i in range(len(img_names)):
        IMG_NAME = img_names[i]
        img = io.imread(Path(jpegdna.__path__[0] +  "/../img/" + IMG_NAME))
        img = img[:8*(img.shape[0]//8), :8*(img.shape[1]//8)]
        #import matplotlib.pyplot as plt
        values = []
        for alpha in [1e-5, 0.145, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            print("==================================")
            print(f"Alpha: {alpha}")
            res = experiment(img, alpha)
            if res is not None:
                if len(res) == 3:
                    compression_rate, PSNR, SSIM_r, MS_SSIM_r, VIF_r = res
                    values.append(value(compression_rate, PSNR, SSIM_r, MS_SSIM_r, VIF_r))
                else:
                    continue
        general_results.append(values)
    with ExcelWriter("results/results_rgb.xlsx") as writer: # pylint: disable=abstract-class-instantiated
        for i in range(len(general_results)):
            dtf = pd.DataFrame(general_results[i])
            dtf.to_excel(writer, sheet_name=img_names[i], index=None, header=True)
