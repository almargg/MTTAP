import torch
import cv2
import numpy as np
from models.Model import GluTracker
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
import imageio
from utils.Visualise import save_gif

train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

batch_size = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = 'cpu'

model = GluTracker().to(device)
model.use_trained_fnet()
model.eval()

for i, (frames, trajs, vsbls, qrs) in enumerate(val_loader):
        frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
        loss = 0
        for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
            images = []
            for j in range(frames.shape[1] - 1):
                startframe = frames[b,0,:,:,:]
                tracked_frame = frames[b,j+1,:,:,:]
                tracked_frames = torch.stack((startframe, tracked_frame), dim=0)
                imgs = model.get_heatmap((0.5, 0.5), tracked_frames)
                cv2.imshow("Heatmap", imgs)
                cv2.waitKey(-1)
                max = np.max(imgs)
                min = np.min(imgs)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                images.append((imgs).astype("uint8").copy())
            save_gif(images)
            