import torch
import tracemalloc
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData, TapvidRgbStacking
from utils.Visualise import create_tap_vid, save_tap_vid
import time


#tracemalloc.start()
#train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/tmp")
train_dataset = KubricDataset("/scratch-second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/tmp")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=8)

davis_data = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")
davis_loader = torch.utils.data.DataLoader(davis_data, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2)
rgb_stack_data = TapvidRgbStacking("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl")
rgb_stack_loader = torch.utils.data.DataLoader(rgb_stack_data, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2)

val_loaders = [davis_loader, rgb_stack_loader]

n_frames = 0

for val_loader in val_loaders:
    for i, (frames, trajs, vsbls, qrs) in enumerate(val_loader):
        n_frames += frames.shape[1]
        if i % 20 == 0:
            print(f"Processed {i} samples in validation dataset.")
        
print(f"Total number of frames in validation datasets: {n_frames}")