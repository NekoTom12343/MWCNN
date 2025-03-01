import torch
import argparse
# import utility
from model.mwcnn import Model
from torch.utils.data import DataLoader
import loss
import os

# Import necessary libraries for TPU
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# import h5py
from option import args
from data.data_provider import SingleLoader, SingleLoader_raw
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
# import model
from torchsummary import summary
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint, MovingAverage, load_checkpoint

def train_fn(rank, args):
    # Set the seed for reproducibility
    torch.manual_seed(args.seed)

    if args.data_type == 'rgb':
        data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    elif args.data_type == 'raw':
        data_set = SingleLoader_raw(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    else:
        print("Data type not valid")
        exit()

    # Use the XLA-compatible DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 workers since TPU handles parallelism differently
    )

    loss_func = loss.CharbonnierLoss()
    device = xm.xla_device()  # Set device to TPU
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

    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            model.model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')

    for epoch in range(start_epoch, args.epochs):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            pred = model(noise)
            loss = loss_func(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)  # Use XLA optimizer step
            average_loss.update(loss)

            if global_step % args.save_every == 0:
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)

            if global_step % args.loss_every == 0:
                print(global_step, "PSNR  : ", calculate_psnr(pred, gt))
                print(average_loss.get_value())
            global_step += 1

        scheduler.step()

if __name__ == "__main__":
    # Launch training on TPU
    xmp.spawn(train_fn, args=(args,), nprocs=8, start_method='fork')
