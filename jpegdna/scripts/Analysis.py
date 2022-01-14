#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 02:11:58 2021

@author: jiaanzhu
"""

from pathlib import Path
from dataclasses import make_dataclass
import math
import pickle
from skimage import io
import pandas as pd
from pandas import ExcelWriter
import jpegdna
from jpegdna.transforms import RGBYCbCr
from IQA_pytorch import SSIM, MS_SSIM, VIFs
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def IQA(img, decoded, code, FORMATTING = False):
    if FORMATTING:
        code_length = 0
        for el in code:
            code_length += len(el)
            compression_rate = 24 * img.shape[0] * img.shape[1] / code_length
    else:
        compression_rate = 24 * img.shape[0] * img.shape[1] / len(code)
        code_length = len(code)
            
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
    D_3 = VIFs(channels=1) # spatial domain VIF
            
    #To get 4-dimension torch tensors, (N, 3, H, W), divide by 255 to let the range between (0,1)
    torch_decoded = torch.FloatTensor(decoded.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
    torch_img = torch.FloatTensor(img.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
    torch_decoded_ycbcr = torch.FloatTensor(decoded_ycbcr.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
    torch_img_ycbcr = torch.FloatTensor(img_ycbcr.astype(int).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)/255
            
    #Calculate SSIM, MS-SSIM, VIF
    #SSIM on luma channel
    #SSIM_value = D_1(torch_decoded_ycbcr[:, [0], :, :] , torch_img_ycbcr[:, [0], :, :], as_loss=False) 
    SSIM_value = D_1(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :] , as_loss=False)
    #MS-SSIM on luma channel
    #MS_SSIM_value = D_2(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :], as_loss=False)
    MS_SSIM_value = D_2(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :] , as_loss=False)
            
    #VIF on spatial domain
    #VIF_value = D_3(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :], as_loss=False)
    VIF_value = D_3(torch_decoded_ycbcr[:, [0], :, :], torch_img_ycbcr[:, [0], :, :] , as_loss=False)

    return code, decoded, compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value

# Logistic fit function
def func(x, a, b, c):
    return a/(1+np.exp(-b*(x-c)))

# Run the code
value = make_dataclass("value", [("Compressionrate", float), ("PSNR", float)
                       ,("SSIM_r", float), ("MS_SSIM_r", float), ("VIF_r", float)])
general_results = []
img_names = []    
for i in range(1, 11):
#for i in range(1, 25):
    img_names.append(f"kodim{i:02d}.png")
for i in range(len(img_names)):
    IMG_NAME = img_names[i]
    print(IMG_NAME)
    img = io.imread(Path(jpegdna.__path__[0] +  "/../img/" + IMG_NAME))
    values = []
    alphas = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    for alpha in alphas:
        decoded_path = f"../../result_img/kodim{i+1:02d}_alpha"+ str(alpha).replace('.','_')+'.png'
        decoded = io.imread(decoded_path)
        DATAFPATH = f"../../result_img/kodim{i+1:02d}_alpha"+ str(alpha).replace('.','_')
        with open(DATAFPATH, 'r', encoding="UTF-8") as f:
            code = f.read()
        code, decoded, compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value = IQA(img, decoded[:,:,0:3], code, FORMATTING = False)
        values.append(value(compression_rate, PSNR, SSIM_value, MS_SSIM_value, VIF_value))
    general_results.append(values)
    
#Plotting
x_standard = np.linspace(1.2,8.4,10)
len_x, len_y = len(img_names), len(x_standard)
PSNR_global, SSIM_global, MS_SSIM_global, VIF_global = np.zeros((len_x, len_y)), np.zeros((len_x, len_y)), np.zeros((len_x, len_y)), np.zeros((len_x, len_y))
    
for i in range(len_x):
    # all the values of image i
    vals = general_results[i]
    lists = [[] for _ in range(5)]
    # compression rate
    for j in range(len(vals)):
        # bitrate = 1/Compression rate
        lists[0].append(1/vals[j].Compressionrate * 24)
        # PSNR
        lists[1].append(vals[j].PSNR)
        # SSIM
        lists[2].append(vals[j].SSIM_r)
        # MS-SSIM
        lists[3].append(vals[j].MS_SSIM_r)
        # VIF
        lists[4].append(vals[j].VIF_r)
        # Plot   
    xdata = lists[0]
        
    plt.figure(f"../../kodim{i+1:02d} Result analysis", dpi = 300, figsize=(16,12))
    plt.subplot(221)
    #ydata = np.array(lists[1])
    ydata = np.array(lists[1])/100 # /100 to avoid overflow
    #plt.plot(xdata, ydata, 'b^', label='data')
    plt.plot(xdata, ydata*100, 'b^', label='data')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    #plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))  
    plt.plot(xdata, func(xdata, *popt)*100, 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))        
    #PSNR_global[i,:] = func(x_standard, *popt)
    PSNR_global[i,:] = func(x_standard, *popt)*100
    plt.xlabel('Rate(nts/pixel)')
    plt.ylabel('PSNR')
    plt.legend()
        
    plt.subplot(222)
    ydata = lists[2]
    plt.plot(xdata, ydata, 'b^', label='data')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    SSIM_global[i,:] = func(x_standard, *popt)
    plt.xlabel('Rate(nts/pixel)')
    plt.ylabel('SSIM')
    plt.legend()
        
    plt.subplot(223)
    ydata = lists[3]
    plt.plot(xdata, ydata, 'b^', label='data')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    MS_SSIM_global[i,:] = func(x_standard, *popt)
    plt.xlabel('Rate(nts/pixel)')
    plt.ylabel('MS-SSIM')
    plt.legend()
        
    plt.subplot(224)
    ydata = lists[4]
    plt.plot(xdata, ydata, 'b^', label='data')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    VIF_global[i,:] = func(x_standard, *popt)
    plt.xlabel('Rate(nts/pixel)')
    plt.ylabel('VIF')
    plt.legend()
    plt.savefig(f"../../results/kodim{i+1:02d}_result.png")

# Plot global
plt.figure("Global pure Result analysis", dpi = 300, figsize=(16,12))
plt.subplot(221)
plt.errorbar(x=x_standard, y=PSNR_global.mean(0), yerr=PSNR_global.std(0), fmt='co--')
plt.xlabel('Rate(nts/pixel)')
#plt.xlabel('Bitrate(nt/bit)')
plt.ylabel('PSNR')

plt.subplot(222)
plt.errorbar(x=x_standard, y=SSIM_global.mean(0), yerr=SSIM_global.std(0), fmt='co--')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('SSIM')

plt.subplot(223)
plt.errorbar(x=x_standard, y=MS_SSIM_global.mean(0), yerr=MS_SSIM_global.std(0), fmt='co--')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('MS-SSIM')

plt.subplot(224)
plt.errorbar(x=x_standard, y=VIF_global.mean(0), yerr=VIF_global.std(0), fmt='co--')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('VIF')      

plt.savefig("../../results/Global_pure_result.png")

#Plotting
x_standard = np.linspace(0, 7, 10)
len_x, len_y = len(img_names), len(x_standard)
PSNR_global, SSIM_global, MS_SSIM_global, VIF_global = np.zeros((len_x, len_y)), np.zeros((len_x, len_y)), np.zeros((len_x, len_y)), np.zeros((len_x, len_y))

lists = [[[] for _ in range(5)] for _ in range(10)]
for i in range(10):
    # all the values of image i
    vals = general_results[i]
    # compression rate
    for j in range(len(vals)):
        # bitrate = 1/Compression rate
        lists[i][0].append(1/vals[j].Compressionrate * 24)
        # PSNR
        lists[i][1].append(vals[j].PSNR)
        # SSIM
        lists[i][2].append(vals[j].SSIM_r)
        # MS-SSIM
        lists[i][3].append(vals[j].MS_SSIM_r)
        # VIF
        lists[i][4].append(vals[j].VIF_r)
        # Plot   
'''
Non-fitted curve
'''        
xdata = lists[0][0]
#Create colors
color = []
for i in range(10):
    color.append(np.random.rand(3,))  
      
plt.figure("Global normal analysis", dpi = 300, figsize=(16,12))

plt.subplot(221)
for i in range(10):
    ydata = np.array(lists[i][1]) # /100 to avoid overflow
    plt.plot(xdata, ydata, c=color[i], label='Image: %5.0f' % (i+1))
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('PSNR')
plt.legend()
        
plt.subplot(222)
for i in range(10):
    ydata = lists[i][2]   
    plt.plot(xdata, ydata, c=color[i], label='Image: %5.0f' % (i+1))
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('SSIM')
plt.legend()
        
plt.subplot(223)
for i in range(10):
    ydata = lists[i][3]
    plt.plot(xdata, ydata, c=color[i], label='Image: %5.0f' % (i+1))
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('MS-SSIM')
plt.legend()
        
plt.subplot(224)
for i in range(10):
    ydata = lists[i][4]
    plt.plot(xdata, ydata, c=color[i], label='Image: %5.0f' % (i+1))
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('VIF')
plt.legend()

plt.savefig("../../results/Global_normal_result.png")

'''
Fitted images
'''

xdata = lists[0][0]
#Create colors
plt.figure("Global all analysis", dpi = 300, figsize=(16,12))

plt.subplot(221)
for i in range(10):
    #ydata = np.array(lists[1])
    ydata = np.array(lists[i][1])/100 # /100 to avoid overflow
    #plt.plot(xdata, ydata, 'b^', label='data')
    #plt.plot(xdata, ydata*100, 'b^', label='data')
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=30000)
    #plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))  
    plt.plot(xdata, func(xdata, *popt)*100, c=color[i],label='Image: %5.0f' % (i+1))        
    #PSNR_global[i,:] = func(x_standard, *popt)
    PSNR_global[i,:] = func(x_standard, *popt)*100
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('PSNR')
plt.legend()
        
plt.subplot(222)
for i in range(10):
    ydata = lists[i][2]    
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), c=color[i],label='Image: %5.0f' % (i+1))
    SSIM_global[i,:] = func(x_standard, *popt)
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('SSIM')
plt.legend()
        
plt.subplot(223)
for i in range(10):
    ydata = lists[i][3]
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), c=color[i],label='Image: %5.0f' % (i+1))
    MS_SSIM_global[i,:] = func(x_standard, *popt)
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('MS-SSIM')
plt.legend()
        
plt.subplot(224)
for i in range(10):
    ydata = lists[i][4]
    popt, pcov = curve_fit(func, xdata, ydata, maxfev=5000)
    plt.plot(xdata, func(xdata, *popt), c=color[i],label='Image: %5.0f' % (i+1))
    VIF_global[i,:] = func(x_standard, *popt)
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('VIF')
plt.legend()

plt.savefig("../../results/Global_all_result.png")






# Plot global
plt.figure("Global Result analysis", dpi=300, figsize=(16,12))
plt.subplot(221)
plt.errorbar(x=x_standard, y=PSNR_global.mean(0), yerr=PSNR_global.std(0), fmt='co--')
plt.plot(lists[1][0][19], lists[1][1][19], 'r^', label='Image 2')
plt.plot(lists[6][0][19], lists[6][1][19], 'g^', label='Image 7')
plt.xlabel('Rate(nts/pixel)')
#plt.xlabel('Bitrate(nt/bit)')
plt.ylabel('PSNR')
plt.legend()

plt.figure("Global Result analysis", figsize=(16,12))
plt.subplot(222)
plt.errorbar(x=x_standard, y=SSIM_global.mean(0), yerr=SSIM_global.std(0), fmt='co--')
plt.plot(lists[1][0][19], lists[1][2][19], 'r^', label='Image 2')
plt.plot(lists[6][0][19], lists[6][2][19], 'g^', label='Image 7')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('SSIM')
plt.legend()

plt.figure("Global Result analysis", figsize=(16,12))
plt.subplot(223)
plt.errorbar(x=x_standard, y=MS_SSIM_global.mean(0), yerr=MS_SSIM_global.std(0), fmt='co--')
plt.plot(lists[1][0][19], lists[1][3][19], 'r^', label='Image 2')
plt.plot(lists[6][0][19], lists[6][3][19], 'g^', label='Image 7')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('MS-SSIM')
plt.legend()

plt.figure("Global Result analysis", figsize=(16,12))
plt.subplot(224)
plt.errorbar(x=x_standard, y=VIF_global.mean(0), yerr=VIF_global.std(0), fmt='co--')
plt.plot(lists[1][0][19], lists[1][4][19], 'r^', label='Image 2')
plt.plot(lists[6][0][19], lists[6][4][19], 'g^', label='Image 7')
plt.xlabel('Rate(nts/pixel)')
plt.ylabel('VIF')    
plt.legend()  

plt.savefig("../../results/Global_result.png")
            
    
with ExcelWriter("../../results/results_rgb.xlsx") as writer: # pylint: disable=abstract-class-instantiated
    for i in range(len(general_results)):
        dtf = pd.DataFrame(general_results[i])
        dtf.to_excel(writer, sheet_name=img_names[i], index=None, header=True)

 
    
 
    
 
    
 