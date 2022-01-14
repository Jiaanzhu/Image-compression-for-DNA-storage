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
from IQA_pytorch import SSIM, MS_SSIM, VIFs
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Choose between "from_img" and "default" for the frequencies
CHOICE = "from_img"
# Enables formatting (if True, bit-rate will be estimated with format taken into account)
FORMATTING = False

def stats(func):
    """Stats printing and exception handling decorator"""

    def inner(*args):
        try:
            code, decoded, res = func(*args)
        except ValueError as err:
            print(err)
        else:
            if FORMATTING:
                code_length = 0
                for el in code:
                    code_length += len(el)
                compression_rate = 24 * img.shape[0] * img.shape[1] / code_length
                print(f"Code length: {code_length}")
            else:
                compression_rate = 24 * img.shape[0] * img.shape[1] / len(code)
                code_length = len(code)
                print(f"Code length: {code_length}")
            
            #Convert RGB to YCbCr
            color_conv = RGBYCbCr()
            img_ycbcr = color_conv.forward(img)
            decoded_ycbcr = color_conv.forward(decoded)
            
            #Calculate MSE and PSNR, Y:U:V = 6:1:1
            MSE_y = ((img_ycbcr[:,:,0].astype(int)-decoded_ycbcr[:,:,0].astype(int))**2).mean()
            MSE_u = ((img_ycbcr[:,:,1].astype(int)-decoded_ycbcr[:,:,1].astype(int))**2).mean()
            MSE_v = ((img_ycbcr[:,:,2].astype(int)-decoded_ycbcr[:,:,2].astype(int))**2).mean()
            PSNR_y = 10 * math.log10((255*255)/MSE_y)
            PSNR_u = 10 * math.log10((255*255)/MSE_u)
            PSNR_v = 10 * math.log10((255*255)/MSE_v)
            PSNR = (PSNR_y * 6 + PSNR_u + PSNR_v)/8
            
            #Call the functions of SSIM, MS-SSIM, VIF
            D_1 = SSIM(channels=1)
            D_2 = MS_SSIM(channels=1)
            D_3 = VIFs(channels=3) # spatial domain VIF
            
            #To get 4-dimension torch tensors, (N, 3, H, W), divide by 255 to let the range between (0,1)
            torch_decoded = torch.FloatTensor(decoded.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
            torch_img = torch.FloatTensor(img.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
            torch_decoded_ycbcr = torch.FloatTensor(decoded_ycbcr.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
            torch_img_ycbcr = torch.FloatTensor(img_ycbcr.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
            
            #Calculate SSIM, MS-SSIM, VIF
            #SSIM on luma channel
            SSIM_value = D_1(torch_decoded_ycbcr[:, [0], :, :] , torch_img_ycbcr[:, [0], :, :], as_loss=False) 
            #MS-SSIM on luma channel
            MS_SSIM_value = D_2(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :], as_loss=False)
            
            #VIF on spatial domain
            VIF_value = D_3(torch_decoded, torch_img, as_loss=False)
            #print(D_3(torch_img, torch_img, as_loss=False))
            #Print out the results
            #print(f"Mean squared error: {MSE}")
            print(f"General PSNR: {PSNR}")
            print(f"SSIM: {SSIM_value}")
            print(f"MS_SSIM: {MS_SSIM_value}")
            print(f"VIF: {VIF_value}")
            print(f"Compression rate: {compression_rate} bits/nt")
            # plt.imshow(decoded)
            # plt.show()
            # io.imsave(str(compression_rate) + ".png", decoded)
            return code, decoded, res, compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value
    return inner

def encode_decode(img, alpha):
    """Function for encoding and decoding"""
    # Coding
    codec = JPEGDNARGB(alpha, formatting=FORMATTING, verbose=False, verbosity=3)
    if CHOICE == "from_img":
        if FORMATTING:
            oligos = codec.full_encode(img, "from_img")
        else:
            (code, res) = codec.full_encode(img, "from_img")
    elif CHOICE == "from_file":
        with open(Path(jpegdna.__path__[0] + "/data/freqs_rgb_4_2_2.pkl"), "rb") as file:
            freqs = pickle.load(file)
        (code, res) = codec.full_encode(img, "from_file", freqs['freq_dc'], freqs['freq_ac'])
    elif CHOICE == "default":
        if FORMATTING:
            oligos = codec.full_encode(img, "default")
        else:
            (code, res) = codec.full_encode(img, "default")
            
    
    # Decoding
    codec2 = JPEGDNARGB(alpha, formatting=FORMATTING, verbose=False, verbosity=3)
    if CHOICE == "from_img":
        if FORMATTING:
            decoded = codec2.full_decode(oligos, "from_img")
        else:
            params = ((res[0][1], res[0][2], res[0][3], res[0][4]),
                      (res[1][1], res[1][2], res[1][3], res[1][4]),
                      (res[1][1], res[2][2], res[2][3], res[2][4]))
            decoded = codec2.full_decode(code, "from_img", params)
    elif CHOICE == "from_file":
        with open(Path(jpegdna.__path__[0] + "/data/freqs_rgb_4_2_2.pkl"), "rb") as file:
            freqs = pickle.load(file)
        params = ((res[0][1], res[0][2], freqs['Y']['freq_dc'], freqs['Y']['freq_ac']),
                  (res[1][1], res[1][2], freqs['Cb']['freq_dc'], freqs['Cb']['freq_ac']),
                  (res[2][1], res[2][2], freqs['Cr']['freq_dc'], freqs['Cr']['freq_ac']))
        decoded = codec2.full_decode(code, "from_file", params)
    elif CHOICE == "default":
        if FORMATTING:
            decoded = codec2.full_decode(oligos, "default")
        else:
            params = ((res[0][1], res[0][2]),
                      (res[1][1], res[1][2]),
                      (res[2][1], res[2][2]))
            decoded = codec2.full_decode(code, "default", params)
    if FORMATTING:
        return oligos, decoded
    return code, decoded, res

@stats
def experiment(img, alpha):
    """Full experiment with stats and exception handling"""
    return encode_decode(img, alpha)

# Logistic fit function
def func(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))

# pylint: disable=missing-function-docstring
if __name__ == '__main__':
    value = make_dataclass("value", [("Compressionrate", float), ("PSNR", float)
                           ,("SSIM_r", float), ("MS_SSIM_r", float), ("VIF_r", float)])
    general_results = []
    img_names = []
    for i in range(1, 11):
    #for i in range(1, 25):
        img_names.append(f"kodim{i:02d}.png")
    for i in range(len(img_names)):
        IMG_NAME = img_names[i]
        img = io.imread(Path(jpegdna.__path__[0] +  "/../img/" + IMG_NAME))
        #img = img[:8*(img.shape[0]//8), :8*(img.shape[1]//8)]
        values = []
        alphas = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        for alpha in alphas:
        #for alpha in [1e-5, 0.145, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            print("==================================")
            print(f"Alpha: {alpha}")
            info = experiment(img, alpha)
            if info is not None:
                if len(info) == 8:
                    code, decoded, res, compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value = info
                    values.append(value(compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value))
                else:
                    continue
            imgoutpath = f"../../result_img/kodim{i+1:02d}_alpha"+ str(alpha).replace('.','_')+'.png'
            io.imsave(imgoutpath, decoded)
            DATAFPATH = f"../../result_img/kodim{i+1:02d}_alpha"+ str(alpha).replace('.','_')
            REQS_OUT_PATH = f"../../result_img/kodim{i+1:02d}_alpha"+ str(alpha).replace('.','_')+'_info'
            
            with open(DATAFPATH, 'w') as f:
                f.write(code)
            with open(REQS_OUT_PATH, "wb") as f:
                    pickle.dump(res, f)
        
        general_results.append(values)
            
    
    with ExcelWriter("../../results/results_rgb.xlsx") as writer: # pylint: disable=abstract-class-instantiated
        for i in range(len(general_results)):
            dtf = pd.DataFrame(general_results[i])
            dtf.to_excel(writer, sheet_name=img_names[i], index=None, header=True)
