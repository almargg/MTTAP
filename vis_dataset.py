import torch
import tracemalloc
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
from utils.Visualise import create_tap_vid, save_tap_vid
import time


#tracemalloc.start()
#train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/tmp")
train_dataset = KubricDataset("/scratch-second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/tmp")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=8)

base_directory = "/scratch_net/biwidl304/amarugg/files/videos"

faulty_samples = 0
start = time.time()
for i, (frames, trajs, vsbls, qrs, depths) in enumerate(train_loader):
    trajs[:,:,:,0] *= frames.shape[4]
    trajs[:,:,:,1] *= frames.shape[3]
    gt = TapData(
            frames,
            trajs,
            vsbls,
            qrs
        )
    N = trajs.shape[2]  # Number of tracks
    print(f"Dataset {i} has {N} tracks")
    if N < 256:
        faulty_samples += 1

    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')
        file_name = f"dataset_{i}.gif"
        path = os.path.join(base_directory, file_name)
        save_tap_vid(gt, path)
    if i == 20:
        break
#end = time.time()
#print(f"Time taken: {end - start} seconds")
print(f"Number of faulty samples: {faulty_samples}")