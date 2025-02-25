import torch
import argparse
import utility
from model.mwcnn import Model
from torch.utils.data import DataLoader
from option import args
from data.data_provider import SingleLoader
from torchsummary import summary
from utils.metric import calculate_psnr, calculate_ssim
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
import math
from PIL import Image
import time

# Check the length of the dataset
data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
print(f"Number of images in dataset: {len(data_set)}")
# Use DataLoader
data_loader = DataLoader(
    data_set,
    batch_size=1,
    shuffle=False,
    num_workers=4
)

# Check the number of batches
num_batches = len(data_loader)
print(f"Number of batches in DataLoader: {num_batches}")

# for epoch in range(10):
#     for step, (noise, gt) in enumerate(data_loader):
#         print(f"Epoch: {epoch}, Step: {step}")
#         noise = noise.to(device)
#         gt = gt.to(device)
#         with torch.no_grad():  # Disable gradient calculation
#             pred = model(noise)
#         pred = pred.detach().cpu()
#         gt = gt.cpu()
#         psnr_t = calculate_psnr(pred, gt)
#         ssim_t = calculate_ssim(pred, gt)
#         print(step, "   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
#         if save_img != '':
#             if not os.path.exists(args.save_img):
#                 os.makedirs(args.save_img)
#             plt.figure(figsize=(15, 15))
#             plt.imshow(np.array(trans(pred[0])))
#             plt.title("denoise KPN DGF " + args.model_type, fontsize=25)
#             image_name = f"image_{step}"
#             plt.axis("off")
#             plt.suptitle(image_name + "   UP   :  PSNR : " + str(psnr_t) + " :  SSIM : " + str(ssim_t), fontsize=25)
#             plt.savefig(os.path.join(args.save_img, image_name + "_" + args.checkpoint + '.png'), pad_inches=0)
#         # Free up memory by deleting variables and clearing the GPU cache
#         del noise, pred, gt
#         torch.cuda.empty_cache()