import torch
import os
import imageio
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Mapping
from dataclasses import dataclass
import random
import cv2

"""
Dataloaders used for all different Datasets
Args:
    data_root: String containing the absolute path to the dataset
Returns:
    rgbs: Video tensor in shape of (S, C, H, W) (int) from [0,255] with H==384 and W==512
    trajs: Trajectory tensor in shape of (S, N, 2) each point is [x,y] (float) from [0,1]^2
    visibles: Visibility tensor in shape of (S, N) (bool) from [0,1]
    query_points: Tensor of query points in shape of (N, 3) each point is [t, x, y] (float) from [0,N-1]x[0,1]^2

"""
VIDEO_INPUT_RESO_CV = (512, 384) # W, H

#cv2 ordering H, W, C
def resize_video(video: torch.Tensor):
    video = video.numpy()
    S, C, H, W = video.shape
    resized_video = np.zeros((S, VIDEO_INPUT_RESO_CV[1], VIDEO_INPUT_RESO_CV[0], C))
    for j in range(video.shape[0]): #S
        frame = video[j,:,:,:]
        frame = np.transpose(frame, (1,2,0))
        r_frame  = cv2.resize(frame, VIDEO_INPUT_RESO_CV, interpolation = cv2.INTER_AREA) 
        resized_video[j] = r_frame
    resized_video = np.transpose(resized_video, ( 0, 3, 1, 2))
    out = torch.from_numpy(resized_video).float()
    return out

def sample_queries_first(
    target_occluded: np.ndarray, #N, S, 
    target_points: np.ndarray, #N, S, 2
    frames: np.ndarray, #S, H, W, C
) -> Mapping[str, np.ndarray]:
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, x, y])) 
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }

@dataclass(eq=False)
class TapData:
    def __init__(self, video, trajectory, visibility, query):
        self.video=video # B, S, C, H, W
        self.trajectory=trajectory # B, S, N, 2
        self.visibility=visibility # B, S, N
        self.query=query # B, N, 3


class KubricDataset(torch.utils.data.DataLoader):
    def __init__(self, data_root, n_traj=128):
        self.data_root = data_root
        assert n_traj < 2048, "To many trajectories"
        self.n_traj = n_traj
        self.seq_names = [
            fname
            for fname in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, fname))
        ]
        print("found %d unique videos in %s" % (len(self.seq_names), data_root))
        
    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        npy_path = os.path.join(self.data_root, seq_name, seq_name + ".npy")
        rgb_path = os.path.join(self.data_root, seq_name, "frames")
        img_paths = sorted(os.listdir(rgb_path))
        rgbs = []
        for i, img_path in enumerate(img_paths):
            rgbs.append(imageio.v2.imread(os.path.join(rgb_path, img_path)))

        rgbs = np.stack(rgbs)
        rgbs = np.transpose(rgbs, (0, 3, 1, 2))
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        trajs = annot_dict["coords"]
        trajs[:,:,0] /= rgbs.shape[2]
        trajs[:,:,1] /= rgbs.shape[3]
        visibility = annot_dict["visibility"]

        n_trajs = trajs.shape[0]
        mask = np.full(n_trajs, False)
        mask[:self.n_traj] = True
        np.random.shuffle(mask)

        trajs = trajs[mask]
        visibility = visibility[mask]

        #TODO ADD data augmentation
        converted = sample_queries_first(visibility, trajs, rgbs)

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2)
        )  # T, N, D
        query_points = torch.from_numpy(converted["query_points"])[0] # T, N
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        rgbs = resize_video(torch.from_numpy(rgbs))
        return rgbs.float(), trajs.float(), visibles.float(), query_points.float()

class TapvidDavisFirst(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        with open(data_root, "rb") as f:
            self.points_dataset = pickle.load(f)
            self.video_names = list(self.points_dataset.keys())
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))
        
    def __len__(self):
        return len(self.points_dataset)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video = self.points_dataset[video_name]
        frames = video["video"]

        target_points = self.points_dataset[video_name]["points"]
        target_occ = self.points_dataset[video_name]["occluded"]
        converted = sample_queries_first(target_occ, target_points, frames)
        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2)
        )  # T, N, D
        #Rescaling
        query_points = torch.from_numpy(converted["query_points"])[0] # T, N
        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2)
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        rgbs = resize_video(rgbs)
        return rgbs.float(), trajs.float(), visibles.float(), query_points.float()
    


class TapvidRgbStacking(torch.utils.data.DataLoader):
    def __init__(self, data_root):
        with open(data_root, "rb") as f:
            self.points_dataset = pickle.load(f)
            self.video_names = [i for i in range(len(self.points_dataset))]
        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))
        
    def __len__(self):
        return len(self.points_dataset)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video = self.points_dataset[video_name]
        frames = video["video"]

        target_points = self.points_dataset[video_name]["points"]
        target_occ = self.points_dataset[video_name]["occluded"]
        converted = sample_queries_first(target_occ, target_points, frames)
        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2)
        )  # T, N, D
        #Rescaling
        query_points = torch.from_numpy(converted["query_points"])[0] # T, N
        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2)
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[
            0
        ].permute(
            1, 0
        )  # T, N
        rgbs = resize_video(rgbs)
        return rgbs.float(), trajs.float(), visibles.float(), query_points.float()
    
if __name__=="__main__":
    def safe_video(frames, traj, vis):
        output_dir = "/scratch_net/biwidl304/amarugg/cotracker/saved_videos/overlays"
        seed = 42
        B, S, C, H, W = frames.shape
        N = traj.shape[2]

        video = frames.detach().cpu()
        trajectory = traj.detach().cpu()
        visibility = vis.detach().cpu()

        random.seed(seed)
        cmap = {
            n: (random.random(), random.random(), random.random())
            for n in range(N)
        }

        for b in range(B):
            for s in range(S):
                frame = video[b, s].permute(1, 2, 0).numpy()  # C, H, W -> H, W, C
                frame = frame / 255
                frame = np.clip(frame, 0, 1)

                plt.figure(figsize=(H / 100, W / 100), dpi=100)
                plt.imshow(frame)
                plt.axis('off')

                for n in range(N):
                    x, y = trajectory[b, s, n]
                    x *= W
                    y *= H
                    is_visible = visibility[b, s, n].item() > 0.5
                    color = cmap[n]

                    plt.scatter(
                        x, y,
                        c=[color],
                        alpha=1.0 if is_visible else 0,
                        s=20,
                        edgecolors='black'
                    )
                plt.savefig(f"{output_dir}/batch{b}_frame{s}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

    #Test dataset
    train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
    val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")
    rgb_stack = TapvidRgbStacking("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    rgb_loader = torch.utils.data.DataLoader(rgb_stack, batch_size=1, shuffle=False)

    for i, (frames, trajs, vsbls, qrs) in enumerate(rgb_loader):

        print("Frames: Shape ", frames.shape, " dtype: ", frames.dtype, " and ranging from ", torch.min(frames), " to ", torch.max(frames))
        print("Trajectories: Shape ", trajs.shape, " dtype: ", trajs.dtype, " and ranging from ", torch.min(trajs), " to ", torch.max(trajs))
        print("Visibles: Shape ", vsbls.shape, " dtype: ", vsbls.dtype, " and ranging from ", torch.min(vsbls), " to ", torch.max(vsbls))
        print("Queries: Shape ", qrs.shape, " dtype: ", qrs.dtype, " and ranging from ", torch.min(qrs), " to ", torch.max(qrs))

        for j in range(qrs.shape[1]):
            qry_coord = [qrs[(0,j,1)], qrs[(0,j,2)]]
            t = qrs[0,j,0].int()
            point_coord = [trajs[(0,t,j,0)], trajs[(0,t, j, 1)]]
            if not(qry_coord[0] == point_coord[0] and qry_coord[1] == point_coord[1]):
                print("ERROR with query point ", qry_coord, " and track ", point_coord)

        if i == 0:
            safe_video(frames, trajs, vsbls)
            
            break