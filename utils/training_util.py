import os
import torch
import shutil
import numbers
import glob

def save_checkpoint(state, is_best, checkpoint_dir, n_iter, max_keep=10):
    filename = os.path.join(checkpoint_dir, "{:06d}.pth.tar".format(n_iter))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_dir,
                                     'model_best.pth.tar'))
    files = sorted(os.listdir(checkpoint_dir))
    rm_files = files[0:max(0, len(files) - max_keep)]
    for f in rm_files:
        os.remove(os.path.join(checkpoint_dir, f))
class MovingAverage(object):
    def __init__(self, n):
        self.n = n
        self._cache = []
        self.mean = 0

    def update(self, val):
        self._cache.append(val)
        if len(self._cache) > self.n:
            del self._cache[0]
        self.mean = sum(self._cache) / len(self._cache)

    def get_value(self):
        return self.mean
def _represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
def load_checkpoint(checkpoint_dir, cuda=True, best_or_latest='best'):
    if best_or_latest == 'best':
        checkpoint_file = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    elif isinstance(best_or_latest, numbers.Number):
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(best_or_latest))
        if not os.path.exists(checkpoint_file):
            files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
            basenames = [os.path.basename(f).split('.')[0] for f in files]
            iters = sorted([int(b) for b in basenames if _represent_int(b)])
            raise ValueError('Available iterations are ({} requested): {}'.format(best_or_latest, iters))
    else:
        files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
        basenames = [os.path.basename(f).split('.')[0] for f in files]
        iters = sorted([int(b) for b in basenames if _represent_int(b)])
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(iters[-1]))
    
    if cuda:
        load_result = torch.load(checkpoint_file, weights_only=True)
    else:
        load_result = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=True)
    
    return load_result

def save_metrics_to_file(psnr_list, ssim_list, checkpoint_dir):
    psnr_file = os.path.join(checkpoint_dir, "average_psnr_list.txt")
    ssim_file = os.path.join(checkpoint_dir, "average_ssim_list.txt")
    
    with open(psnr_file, "w") as f:
        for epoch, psnr in psnr_list:
            f.write(f"{epoch},{psnr}\n")
    
    with open(ssim_file, "w") as f:
        for epoch, ssim in ssim_list:
            f.write(f"{epoch},{ssim}\n")

def load_metrics_from_file(checkpoint_dir):
    psnr_file = os.path.join(checkpoint_dir, "average_psnr_list.txt")
    ssim_file = os.path.join(checkpoint_dir, "average_ssim_list.txt")
    
    psnr_list = []
    ssim_list = []
    
    if os.path.exists(psnr_file):
        with open(psnr_file, "r") as f:
            for line in f:
                epoch, psnr = line.strip().split(",")
                psnr_list.append([int(epoch), float(psnr)])
    
    if os.path.exists(ssim_file):
        with open(ssim_file, "r") as f:
            for line in f:
                epoch, ssim = line.strip().split(",")
                ssim_list.append([int(epoch), float(ssim)])
    
    return psnr_list, ssim_list