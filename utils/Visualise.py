import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image
from dataset.Dataloader import TapData
import torch


def create_tap_vid(sample: TapData, idx=0):
    """
    Create a video from a TapData sample with trajectory points overlayed.
    """
    video = sample.video.numpy()[idx] # (S, C, H, W)[0,255]
    video = np.moveaxis(video, 1, -1) # (S, H, W, C)
    trajectory = sample.trajectory.numpy()[idx].astype(int) # (S, N, 2)
    visibility = sample.visibility.numpy()[idx] # (S, N)
    qrs = sample.query.numpy()[idx] # (N, 3)

    circle_thickness = 2
    circle_radius = 5
    
    S, H, W, C = video.shape
    N = trajectory.shape[1]

    colors = generate_colors(N) 

    # Paint the trajectory points on the video
    out = video.copy().astype(np.uint8)  
    for s in range(S):
        for n in range(N):
            #Only draw tracked points
            if qrs[n,0] > s:
                continue
            x, y = trajectory[s, n]
            if visibility[s, n] > 0.5:  
                thickness = -1  
            else:
                thickness = circle_thickness 
            cv2.circle(out[s], (x,y), circle_radius, colors[n], thickness)

        
    # Reduce the number of colors to 254 for GIF compatibility
    for s in range(S):
        frame = Image.fromarray(out[s])
        if s == 0:
            frame = frame.quantize(colors=254, method=Image.Quantize.MAXCOVERAGE)
            first_img = frame
            colors = np.array(first_img.getpalette()).reshape(-1, 3)
            frame = np.array(frame)
            for c in range(colors.shape[0]):
                out[s][frame == c] = colors[c]
        else:
            frame = frame.quantize(colors=254, palette=first_img, method=Image.Quantize.MAXCOVERAGE)
            frame = np.array(frame)
            for c in range(colors.shape[0]):
                out[s][frame == c] = colors[c]

    out = np.moveaxis(out, -1, 1)
    out = torch.from_numpy(out)
    return out / 255.0  # Normalize to [0, 1] range

def generate_colors(n):
    colors = []
    for i in range(n):
        hue = (240*i) / (n-1)
        hsv_color = np.uint8([[[hue * 179 // 360, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])))
    return colors


def paint_frame(frame, points, colors, visibility, qrs, s):
    circle_radius = 5
    circle_thickness = 2
    N = points.shape[0]
    for n in range(N):
        if qrs[n, 0] > s:
            continue
        x, y = points[n]
        if visibility[n] > 0.5:
            thickness = -1  # Filled circle
        else:
            thickness = circle_thickness  # Outline circle
        cv2.circle(frame, (x, y), circle_radius, colors[n], thickness)
    return frame

def vis_against_gt(sample: TapData, gt: TapData, idx=0):
    """
    Visualize the predicted trajectory against the ground truth.
    """
    video = sample.video.numpy()[idx]  # (S, C, H, W)[0,255]
    video = np.moveaxis(video, 1, -1)  # (S, H, W, C)
    pred_trajectory = sample.trajectory.numpy()[idx].astype(int)  # (S, N, 2)
    pred_visibility = sample.visibility.numpy()[idx]  # (S, N)
    gt_trajectory = gt.trajectory.numpy()[idx].astype(int)  # (S, N_gt, 2)
    gt_visibility = gt.visibility.numpy()[idx]  # (S, N_gt)

    qrs = sample.query.numpy()[idx]  # (N, 3)

    
    S, H, W, C = video.shape
    N = gt_trajectory.shape[1]

    colors = generate_colors(N) 

    pred = video.copy().astype(np.uint8)  
    gt = video.copy().astype(np.uint8)
    for s in range(S):
        pred[s] = paint_frame(pred[s], pred_trajectory[s], colors, pred_visibility[s], qrs, s)
        gt[s] = paint_frame(gt[s], gt_trajectory[s], colors, gt_visibility[s], qrs, s)
    concat = np.concatenate((pred, gt), axis=2)  # Concatenate along width

    # Display with cv2
    for s in range(S):
        cv2.imshow('Predicted vs Ground Truth', cv2.cvtColor(concat[s], cv2.COLOR_RGB2BGR))
        cv2.waitKey(-1)

def display_corr_volume(corr_volume):
    """
    Display the correlation volume as a grid of images.
    """
    N, S, S, S, S = corr_volume.shape

    center = S // 2
    




