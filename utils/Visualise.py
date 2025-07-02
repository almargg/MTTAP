import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
import imageio
from PIL import Image
import os
from dataset.Dataloader import TapData
import torch


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

def paint_points(frame, coordinates):
    circle_radius = 10
    circle_color = (0, 255, 0)
    circle_thickness = 2
    
    for i in range(coordinates.shape[0]):
        x, y = int(coordinates[i,0]), int(coordinates[i,1])
        cv2.circle(frame, (x,y), circle_radius, circle_color, circle_thickness)

    return frame

def display_torch_frame(frame, coordinates=None, visibles=None):
    frame = frame.numpy()
    frame = np.moveaxis(frame, 0, -1)
    frame = (frame).astype("uint8")
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if coordinates != None:
        frame = paint_points(frame, coordinates)
    cv2.imshow("Frame", frame)
    cv2.waitKey(-1)

def paint_frame(frame, points, visibility, colors):
    circle_radius = 5
    circle_thickness = 2

    for i in range(points.shape[0]):
        x, y = int(points[i,0]), int(points[i,1])
        if visibility[i] > 0.5:
            thickness = -1
        else:
            thickness = circle_thickness
        cv2.circle(frame, (x,y), circle_radius, colors[i], thickness)
    return frame

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = (240*i) / (n-1)
        hsv_color = np.uint8([[[hue * 179 // 360, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    return colors

def save_tap_vid(data: TapData, out_path):
    base_size = 600

    video = data.video.numpy()[0]
    video = np.moveaxis(video, 1, -1) # (S, H, W, C)
    trajectory = data.trajectory.numpy()[0].astype(int) # (S, N, 2)
    visibility = data.visibility.numpy()[0] # (S, N)
    S, N = visibility.shape

    images = []
    colors = generate_colors(N)

    for i in range(S):
        frame = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        points = trajectory[i]
        vsbls = visibility[i]
        #resize frame and points
        new_frame = paint_frame(frame.copy(), points, vsbls, colors)

        new_frame = np.uint8(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
        new_frame = Image.fromarray(new_frame)
        if i == 0:
            new_frame = new_frame.quantize(colors=254, method=Image.Quantize.MAXCOVERAGE)
            first_image = new_frame
        else:
            new_frame = new_frame.quantize(colors=254, palette=first_image, method=Image.Quantize.MAXCOVERAGE)
        images.append(new_frame)
    images[0].save(out_path, format="GIF", save_all=True, append_images=images[1:], duration=80, loop=0)
    


