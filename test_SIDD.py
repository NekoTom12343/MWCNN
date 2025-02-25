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

torch.set_num_threads(4)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.manual_seed(0)

def test(args):
    model = Model(args)
    save_img = args.save_img
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_iter']
        state_dict = checkpoint['state_dict']
        model.model.load_state_dict(state_dict)
        print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    except Exception as e:
        print('=> no checkpoint file to be loaded. Error:', e)
        exit(1)
    model.eval()
    trans = transforms.ToPILImage()
    torch.manual_seed(0)

    # Use SingleLoader and DataLoader
    data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    for epoch in range(10):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            with torch.no_grad():  # Disable gradient calculation
                pred = model(noise)
            pred = pred.detach().cpu()
            gt = gt.cpu()
            psnr_t = calculate_psnr(pred, gt)
            ssim_t = calculate_ssim(pred, gt)
            print(step, "   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
            if save_img != '':
                if not os.path.exists(args.save_img):
                    os.makedirs(args.save_img)
                plt.figure(figsize=(15, 15))
                plt.imshow(np.array(trans(pred[0])))
                plt.title("denoise KPN DGF " + args.model_type, fontsize=25)
                image_name = f"image_{step}"
                plt.axis("off")
                plt.suptitle(image_name + "   UP   :  PSNR : " + str(psnr_t) + " :  SSIM : " + str(ssim_t), fontsize=25)
                plt.savefig(os.path.join(args.save_img, image_name + "_" + args.checkpoint + '.png'), pad_inches=0)
            # Free up memory by deleting variables and clearing the GPU cache
            del noise, pred, gt
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test(args)