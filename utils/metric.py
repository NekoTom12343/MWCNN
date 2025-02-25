import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    while len(tensor.size()) < 4:
        tensor = tensor.unsqueeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()

def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += peak_signal_noise_ratio(target_tf[im_idx, ...],
                                        output_tf[im_idx, ...],
                                        data_range=255)
        n += 1.0
    return psnr / n

def calculate_ssim(output_img, target_img, win_size=7):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim_value = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim_value += structural_similarity(target_tf[im_idx, ...],
                                            output_tf[im_idx, ...],
                                            multichannel=True,
                                            data_range=255,
                                            win_size=win_size,
                                            channel_axis=2)  # Assuming the channel axis is the last axis
        n += 1.0
    return ssim_value / n