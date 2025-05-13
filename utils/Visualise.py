import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
import imageio
import os
from dataset.Dataloader import TapData



def create_tap_vid(sample: TapData, idx=0):
    video = sample.video.numpy()[idx] # (S, C, H, W)
    video = np.moveaxis(video, 1, -1) # (S, H, W, C)
    trajectory = sample.trajectory.numpy()[idx].astype(int) # (S, N, 2)
    visibility = sample.visibility.numpy()[idx] # (S, N)
    qrs = sample.query.numpy()[idx] # (N, 3)

    #max = np.max(video)
    #min = np.min(video)

    alpha = 0.5
    radius = 3
    thickness = -1
    seed = 42
    S, H, W, C = video.shape
    N = trajectory.shape[1]

    random.seed(seed)
    cmap = {
        n: (random.random(), random.random(), random.random())
        for n in range(N)
    }

    out = video.copy() / 255
    for s in range(S):
        overlay = video[s].copy() / 255
        for n in range(N):
            #Only draw tracked points
            if qrs[n,0] > s:
                continue
            if visibility[s,n] > 0.5:
                cv2.circle(out[s], (trajectory[s,n,0], trajectory[s,n,1]), radius, cmap[n], thickness)
                cv2.circle(overlay, (trajectory[s,n,0], trajectory[s,n,1]), radius, cmap[n], thickness)
            else:
                cv2.circle(overlay, (trajectory[s,n,0], trajectory[s,n,1]), radius, cmap[n], thickness)
        out[s] = cv2.addWeighted(overlay, alpha, out[s], 1 - alpha, 0)

    out = np.moveaxis(out, -1, 1)
    out = torch.from_numpy(out)
    return out
        
def save_gif(images):
    save_dir = "/scratch_net/biwidl304/amarugg/gluTracker/media"
    imageio.mimsave(os.path.join(save_dir, "movie.gif"), images, fps=20)
    print("GIF saved")

def display_tap_vid(sample: TapData, idx=0, save=False):
    
    images = []
    video = create_tap_vid(sample, idx=idx)
    for s in range(video.shape[0]):
        frame = video[s]
        frame = np.moveaxis(frame.numpy(), 0, -1)
        images.append((frame * 255).astype("uint8").copy())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Video", frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    if save:
        save_gif(images)

