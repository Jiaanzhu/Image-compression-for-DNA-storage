"""Jpeg DNA evaluation script"""

from pathlib import Path
from dataclasses import make_dataclass
import math
import pickle
from skimage import io
import pandas as pd
from pandas import ExcelWriter
import jpegdna
from jpegdna.codecs import JPEGDNA
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
            compression_rate = 8 * img.shape[0] * img.shape[1] / len(code)            
            
            #Calculate MSE and PSNR
            MSE = ((img.astype(int)-decoded.astype(int))**2).mean()
            PSNR = 10 * math.log10((255*255)/MSE)
            
            #Call the functions of SSIM, MS-SSIM, VIF
            D_1 = SSIM()
            D_2 = MS_SSIM()
            #D_3 = VIF()
            
            #To get 4-dimension torch tensors
            torch_decoded = torch.FloatTensor(decoded).unsqueeze(0).unsqueeze(0)
            torch_img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
            
            #Calculate SSIM, MS-SSIM, VIF
            SSIM_r = D_1(torch_decoded , torch_img, as_loss=False)
            MS_SSIM_r = D_2(torch_decoded, torch_img)
            #VIF_r = D_3(torch_decoded, torch_img)
            
            #Print out the results
            print(f"Mean squared error: {MSE}")
            print(f"PSNR: {PSNR}")
            print(f"SSIM: {SSIM_r}")
            print(f"MS_SSIM_r: {MS_SSIM_r}")
            #print(f"VIF: {VIF_r}")
            print(f"Compression rate: {compression_rate} bits/nt")
            # io.imsave(str(compression_rate) + ".png", decoded)
            return compression_rate, PSNR, SSIM_r, MS_SSIM_r
    return inner

def encode_decode(img, alpha):
    """Function for encoding and decoding"""
    choice = "from_img"
    codec = JPEGDNA(alpha, verbose=False, verbosity=3)
    if choice == "from_img":
        (code, res) = codec.full_encode(img, "from_img")
    elif choice == "from_file":
        with open(Path(jpegdna.__path__[0] + "/data/freqs.pkl"), "rb") as file:
            freqs = pickle.load(file)
        (code, res) = codec.full_encode(img, "from_file", freqs['freq_dc'], freqs['freq_ac'])
    codec2 = JPEGDNA(alpha, verbose=False, verbosity=3)
    decoded = codec2.full_decode(code, res[1], res[2], res[3], res[4])
    return code, decoded

@stats
def experiment(img, alpha):
    """Full experiment with stats and exception handling"""
    return encode_decode(img, alpha)

if __name__ == '__main__':
    value = make_dataclass("value", [("Compressionrate", float), ("PSNR", float)
                                     ,("SSIM_r", float), ("MS_SSIM_r", float)])
    general_results = []
    img_names = ["kodim_gray_1.png"]
    #img_names = ["1_gray.jpg", "2_gray.jpg", "3_gray.jpg", "4_gray.jpg", "5_gray.jpg", "6_gray.jpg", "7_gray.jpg", "8_gray.jpg"]
    for i in range(len(img_names)):
        IMG_NAME = img_names[i]
        img = io.imread("img/" + IMG_NAME)
        img = img[:8*(img.shape[0]//8), :8*(img.shape[1]//8)]
        # import matplotlib.pyplot as plt
        values = []
        for alpha in [1e-5, 0.145, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        #for alpha in [0.5]:
            print("==================================")
            print(f"Alpha: {alpha}")
            res = experiment(img, alpha)
            if res is not None:
                compression_rate, PSNR, SSIM_r, MS_SSIM_r = res
                values.append(value(compression_rate, PSNR, SSIM_r, MS_SSIM_r))
        general_results.append(values)
   #with ExcelWriter("res/results.xlsx") as writer: # pylint: disable=abstract-class-instantiated
    #    for i in range(len(general_results)):
     #       dtf = pd.DataFrame(general_results[i])
      #      dtf.to_excel(writer, sheet_name=img_names[i], index=None, header=True) 
