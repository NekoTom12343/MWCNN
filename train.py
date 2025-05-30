import torch
import argparse
# import utility
from model.mwcnn import Model
from torch.utils.data import DataLoader
import loss
import os

# import h5py
from option import args
from data.data_provider import SingleLoader,SingleLoader_raw
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
# import model
from torchsummary import summary
from utils.metric import calculate_psnr, calculate_ssim
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint, load_metrics_from_file, save_metrics_to_file
# from collections import OrderedDict


if __name__ == "__main__":

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    # checkpoint = utility.checkpoint(args)
    if args.data_type =='rgb':
        data_set = SingleLoader(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size)
    elif args.data_type == 'raw':
        data_set = SingleLoader_raw(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size)
    else:
        print("Data type not valid")
        exit()
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # loss_func = loss.Loss(args,None)
    loss_func = loss.CharbonnierLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = Model(args).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 10, 12, 14, 16], 0.8)

    optimizer.zero_grad()
    global_step = 0
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        average_psnr_list = []
        average_ssim_list = []
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            average_psnr_list, average_ssim_list = load_metrics_from_file(checkpoint_dir)  # Load metrics from files
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "model."+ k  # remove `module.`
            #     new_state_dict[name] = v
            model.model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            average_psnr_list = []
            average_ssim_list = []
            print('=> no checkpoint file to be loaded.')

    psnr_sum = 0
    ssim_sum = 0
    loss_every_count = 0
    for epoch in range(start_epoch, args.epochs):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            pred = model(noise)
            # print(pred.size())
            loss = loss_func(pred,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)

            # Calculate PSNR and SSIM for the current step
            psnr = calculate_psnr(pred, gt)
            ssim = calculate_ssim(pred, gt)

            # Update the running sums and count
            psnr_sum += psnr
            ssim_sum += ssim
            del psnr, ssim
            loss_every_count += 1
            if global_step % args.loss_every == 0:
                avg_psnr = psnr_sum / loss_every_count if loss_every_count > 0 else 0
                avg_ssim = ssim_sum / loss_every_count if loss_every_count > 0 else 0
                print(global_step, "Average PSNR:", avg_psnr,"  |  ", "Average SSIM:", avg_ssim)  
                print(average_loss.get_value())
                if global_step % args.save_every != 0:
                    psnr_sum = 0
                    ssim_sum = 0
                    loss_every_count = 0
            if global_step % args.save_every == 0:
                print(len(average_loss._cache))
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False
                # Calculate the average PSNR and SSIM for the current epoch
                avg_psnr_epoch = psnr_sum / loss_every_count if loss_every_count > 0 else 0
                avg_ssim_epoch = ssim_sum / loss_every_count if loss_every_count > 0 else 0

                # Append the epoch-level averages to the lists
                average_psnr_list.append([epoch, avg_psnr_epoch])
                average_ssim_list.append([epoch, avg_ssim_epoch])

                # Reset the running sums and count for the next epoch
                psnr_sum = 0
                ssim_sum = 0
                loss_every_count = 0
                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
                save_metrics_to_file(average_psnr_list, average_ssim_list, checkpoint_dir)
            global_step +=1
        scheduler.step()

    # print(model)
    # print(summary(model,[(3,512,512),[8]]))