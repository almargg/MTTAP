import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Modules import BasicEncoder, PredictionMLP
import cv2
import numpy as np


class GluTracker(nn.Module):
    def __init__(self, n_corr_lvl=2, r_win=1, stride=4, latent_dim=128, save_dir = "/scratch_net/biwidl304/amarugg/gluTracker/weights"):
        super().__init__()
        self.fnet = BasicEncoder(stride=stride)
        print(f"Feature Net consisting of {sum(p.numel() for p in self.fnet.parameters())} Parameters")
        self.r_win = r_win
        self.win_size = 2 * r_win + 1
        self.predictor = PredictionMLP(self.win_size*self.win_size)
        print(f"Predictor consiting of {sum(p.numel() for p in self.predictor.parameters())} Parameters")
        self.stride = stride
        self.n_corr_lvl = n_corr_lvl
        self.latent_dim = latent_dim
        self.safe_dir = save_dir


    def save(self):
        torch.save(self.state_dict(), self.safe_dir + "/gluTracker.pth")

    def load(self):
        self.load_state_dict(torch.load(self.safe_dir + "/gluTracker.pth", weights_only=True))

    def use_trained_fnet(self):
        self.fnet.load_state_dict(torch.load(self.safe_dir + "/fnet.pth", weights_only=True))
        for param in self.fnet.parameters():
            param.requires_grad = False

    """
    fmaps: (B, S, C, H, W )
    points: (B, N, 2)

    returns:
        features: (B, N, 2*r+1, 2*r+1)
    """
    def display_heatmap(self, position, frames):
        N, _, _, _ = frames.shape
        r = 1
        assert N == 2

        fmaps = self.fnet(frames)
        _, C, H, W = fmaps.shape
        x = int(position[0] * fmaps.shape[3])
        y = int(position[1] * fmaps.shape[2])
        template = fmaps[0, :, y-r:y+r+1, x-r:x+r+1] / (2*r+1)**2
        target = fmaps[1,:,:,:]
        template = F.normalize(template, p=2, dim=0)  
        target = F.normalize(target, p=2, dim=0)   

        template = template.unsqueeze(0)
        target = target.unsqueeze(0)
        score = F.conv2d(target, template, padding=r) / (2*r+1)**2
        heatmap = score.squeeze(0).squeeze(0).numpy()
        heatmap = (heatmap + 1) / 2
        heatmap *= 255
        best_match = np.argmax(heatmap)
        x_best = (best_match % heatmap.shape[1])*4
        y_best = (best_match // heatmap.shape[1])*4

        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (heatmap.shape[1]*4, heatmap.shape[0]*4))

        frames = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)

        frames[0, y*4-1:y*4+2, x*4-1:x*4+2, :] = np.array([0,0,255])
        frames[1, y_best-1:y_best+2, x_best-1:x_best+2, :] = np.array([255, 0, 0])
        imgs = np.concatenate((frames[0], frames[1], heatmap), axis=1)
        cv2.imshow("Heatmap", imgs)
        cv2.waitKey(-1)

    def sample_features(self, fmap, points):
        B, C, H, W = fmap.shape
        _, N, _ = points.shape

        device = points.device

        lin_offset = torch.linspace(-self.r_win, self.r_win, steps=self.win_size, device=device)
        dy, dx = torch.meshgrid(lin_offset, lin_offset, indexing="ij")
        offset_grid = torch.stack((dx, dy), dim=-1).view(-1, 2) # (2*r+1)^2, 2

        #Normalise to [-1; 1]
        offset_grid[:, 0] = offset_grid[:, 0] / (W - 1) * 2
        offset_grid[:, 1] = offset_grid[:, 1] / (H - 1) * 2

        points = points * 2 - 1

        grid = points.unsqueeze(2) + offset_grid.view(1,1, self.win_size*self.win_size, 2) # B, N, (2*r+1)^2, 2
        grid = grid.unsqueeze(1) # B, N, (2*r+1)^2, 2
        grid = grid.view(B, N*self.win_size*self.win_size, 1, 2)

        fmap = fmap.view(B, C, H, W) # B, C, H, W

        sampled = F.grid_sample(fmap, grid, align_corners=True, mode="bilinear", padding_mode="zeros") # B*S, C, N*(2*r+1)^2,1
        sampled = sampled.view(B, C, N, self.win_size, self.win_size)
        
        return sampled
    

    def resize_feature_map(self, fmap, target_size=(108,135)):
        B, S, C, _, _ = fmap.shape
        fmap = fmap.view(B * S, C, 96, 128)
        resized = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
        return resized.view(B, S, C, *target_size)

    def extract_fmaps_pyramids(self, frames):
        B, S, C, H, W = frames.shape
        dtype = frames.dtype

        #Normalise video
        frames = 2 * (frames / 255) - 1 # B, S, 3, 384, 512

        fmap = self.fnet(frames.reshape(B*S, C, H, W)) # N, 128, 96, 128
        fmap = fmap.reshape(B, S, self.latent_dim, H // self.stride, W // self.stride) #B, S, 128, 96, 128
        fmap = fmap.to(dtype)

        #Rescale fmap to have h/w that is multiple of win size
        fmap = self.resize_feature_map(fmap)

        #Build feature Pyramids
        fmaps_pyramid = []
        fmaps_pyramid.append(fmap)
        for i in range(self.n_corr_lvl - 1):
            fmap_ = fmap.reshape(B*S, self.latent_dim, fmap.shape[-2], fmap.shape[-1])
            fmap_ = F.avg_pool2d(fmap_, kernel_size=self.win_size, stride=self.win_size)
            fmap = fmap_.reshape(B, S, self.latent_dim, fmap.shape[-2] // self.win_size, fmap.shape[-1] // self.win_size)
            fmaps_pyramid.append(fmap)

        return fmaps_pyramid
    
    def create_corr_vol(self, fmaps, queries, pred_pos):
        features = self.sample_features(fmaps[:,0,:,:,:], queries) # B, C, N, (2*r+1)^2
        feature_matches = self.sample_features(fmaps[:,1,:,:,:], pred_pos) # B, C, N, (2*r+1)^2

        #From LocoTrack
        """
        rename to query features and tracking features
        corr_4D
        """
        B, C, N, H, W = features.shape
        _, _, _, H_, W_ = feature_matches.shape

        # Reshape to (B, C, N, H*W)
        features_flat = features.view(B, C, N, H * W)
        # Reshape to (B, C, N, H_*W_)
        feature_matches_flat = feature_matches.view(B, C, N, H_ * W_)

        # Normalize along the channel dimension
        features_norm = F.normalize(features_flat, p=2, dim=1)  # (B, C, N, H*W)
        feature_matches_norm = F.normalize(feature_matches_flat, p=2, dim=1)  # (B, C, N, H_*W_)

        # Compute cosine similarity: (B, N, H*W, H_*W_)
        corr = torch.einsum('bcnh, bcnk -> bnhk', features_norm, feature_matches_norm)

        # Reshape to (B, N, H, W, H_, W_)
        corr_vol = corr.view(B, N, H, W, H_, W_)
        corr_vol = corr_vol.reshape(B*N, self.win_size**4)

        return corr_vol
    
    def calculate_position_update(self, pred, H, W):
        device = pred.device
        pred_x = pred.sum(dim=-2) #sum along y axis to have all values with same x coordinate
        pred_y = pred.sum(dim=-1) 

        pixel_dist_x = 1 / (W - 1)
        pixel_dist_y = 1 / (H - 1)

        pixel_dsts_x = torch.linspace(-self.r_win*pixel_dist_x, self.r_win*pixel_dist_x, self.win_size, device=device)
        pixel_dsts_y = torch.linspace(-self.r_win*pixel_dist_y, self.r_win*pixel_dist_y, self.win_size, device=device)

        d_x = pred_x @ pixel_dsts_x
        d_y = pred_y @ pixel_dsts_y

        return d_x, d_y

    
    def train_video(self, qrs, frames, trajs, visibility, loss_f):
        self.train()
        B, S, C, H, W = frames.shape
        fmaps_pyramid = self.extract_fmaps_pyramids(frames)
        loss = 0
        iter = 0
        for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
            for j in range(frames.shape[1] - 1):
                trajs_ = trajs[b, j, :, :] # N, 2
                qrs_msk = qrs[b,:,0] <= j #N
                queries = trajs_[qrs_msk][None, :, :] #Ground Truth position of all trackable trajectories

                gt_traj = trajs[b, j+1, :, :][qrs_msk]
                d_traj = gt_traj - queries
                gt_vis = visibility[b, j+1, :][qrs_msk]
                
                pred_pos = queries # B, N, 2
                N = queries.shape[1]
                for i in range(self.n_corr_lvl):
                    fmaps = fmaps_pyramid[self.n_corr_lvl - i - 1][:, j:j+2]
                    corr_vol = self.create_corr_vol(fmaps, queries, pred_pos)

                    pred, vis = self.predictor(corr_vol)
                    pred = pred.reshape(1, N, self.win_size, self.win_size)
                    vis = vis.reshape(1, N)
                    B, S, C,  H, W = fmaps.shape
                    loss += loss_f(pred[0], vis[0], d_traj, gt_vis, H, W)
                    iter += 1

                    d_x, d_y = self.calculate_position_update(pred, H, W)

                    pred_pos[:,:,0] += d_x
                    pred_pos[:,:,1] += d_y
        loss.backward()

        return loss.item() / iter



            
    def forward(self, queries, frames):
        B, S, C, H, W = frames.shape
        device = frames.device
        B, N, _ = queries.shape
        fmaps_pyramid = self.extract_fmaps_pyramids(frames)

        pred_pos = queries # B, N, 2
        pred_vis = torch.zeros(B, N, device=device) # B, N
        #Coarse to fine search
        for i in range(self.n_corr_lvl):

            corr_vol = self.create_corr_vol(fmaps_pyramid[self.n_corr_lvl - i - 1], queries, pred_pos)

            pred, vis = self.predictor(corr_vol)
            pred = pred.reshape(B, N, self.win_size, self.win_size)
            vis = vis.reshape(B,N)

            d_x, d_y = self.calculate_position_update(pred, fmaps_pyramid[self.n_corr_lvl - i - 1].shape[3], fmaps_pyramid[self.n_corr_lvl - i - 1].shape[4])

            pred_pos[:,:,0] += d_x
            pred_pos[:,:,1] += d_y

            pred_vis[:,:] += vis
        
        pred_vis /= self.n_corr_lvl

        return pred_pos, pred_vis