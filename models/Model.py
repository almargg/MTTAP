import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Modules import BasicEncoder, PredictionMLP
import os


class GluTracker(nn.Module):
    def __init__(self, n_corr_lvl=2, r_win=1, stride=4, latent_dim=128, save_dir = "/scratch_net/biwidl304/amarugg/gluTracker/weights/gluTracker.pth"):
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
        torch.save(self.state_dict(), self.safe_dir)

    def load(self):
        self.load_state_dict(torch.load(self.safe_dir, weights_only=True))

    """
    fmaps: (B, S, C, H, W )
    points: (B, N, 2)

    returns:
        features: (B, N, 2*r+1, 2*r+1)
    """
    def sample_features(self, fmaps, points):
        B, C, H, W = fmaps.shape
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

        fmaps = fmaps.view(B, C, H, W) # B, C, H, W

        sampled = F.grid_sample(fmaps, grid, align_corners=True, mode="bilinear", padding_mode="zeros") # B*S, C, N*(2*r+1)^2,1
        sampled = sampled.view(B, C, N, self.win_size, self.win_size)

        return sampled

    def forward(self, queries, frames):
        B, S, C, H, W = frames.shape
        device = frames.device
        B, N, _ = queries.shape
        dtype = frames.dtype

        #Normalise video
        frames = 2 * (frames / 255) - 1 # B, S, 3, 384, 512

        fmaps = self.fnet(frames.reshape(B*S, C, H, W)) # N, 128, 96, 128
        fmaps = fmaps.reshape(B, S, self.latent_dim, H // self.stride, W // self.stride) #B, S, 128, 96, 128
        fmaps = fmaps.to(dtype)
        #Rescale fmaps to have h/w that is multiple of win size

        #Build feature Pyramids
        fmaps_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.n_corr_lvl - 1):
            fmaps_ = fmaps.reshape(B*S, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1])
            fmaps_ = F.avg_pool2d(fmaps_, kernel_size=self.win_size, stride=self.win_size)
            fmaps = fmaps_.reshape(B, S, self.latent_dim, fmaps.shape[-2] // self.win_size, fmaps.shape[-1] // self.win_size)
            fmaps_pyramid.append(fmaps)

        pred_pos = queries # B, N, 2
        pred_vis = torch.zeros(B, N, device=device) # B, N
        #Coarse to fine search
        for i in range(self.n_corr_lvl):
            #Create correlation volumes
            features = self.sample_features(fmaps_pyramid[self.n_corr_lvl - i - 1][:,0,:,:,:], queries) # B, C, N, (2*r+1)^2
            feature_matches = self.sample_features(fmaps_pyramid[self.n_corr_lvl - i - 1][:,1,:,:,:], pred_pos) # B, C, N, (2*r+1)^2

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

            pred, vis = self.predictor(corr_vol)
            pred = pred.reshape(B, N, self.win_size, self.win_size)
            vis = vis.reshape(B,N)

            pred_x = pred.sum(dim=-1) #sum along y axis to have all values with same x coordinate
            pred_y = pred.sum(dim=-2) 

            pixel_dist_x = 1 / (fmaps_pyramid[self.n_corr_lvl - i - 1].shape[4] - 1)
            pixel_dist_y = 1 / (fmaps_pyramid[self.n_corr_lvl - i - 1].shape[3] - 1)

            pixel_dsts_x = torch.linspace(-self.r_win*pixel_dist_x, self.r_win*pixel_dist_x, self.win_size, device=device)
            pixel_dsts_y = torch.linspace(-self.r_win*pixel_dist_y, self.r_win*pixel_dist_y, self.win_size, device=device)

            d_x = pred_x @ pixel_dsts_x
            d_y = pred_y @ pixel_dsts_y

            pred_pos[:,:,0] += d_x
            pred_pos[:,:,1] += d_y

            pred_vis[:,:] += vis
        
        pred_vis /= self.n_corr_lvl

        return pred_pos, pred_vis