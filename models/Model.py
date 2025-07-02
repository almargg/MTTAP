import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Modules import BasicEncoder
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class query():
    coordinates: torch.Tensor
    anchors: torch.Tensor

class GluTracker(nn.Module):
    def __init__(self, n_corr_lvl=2, r_win=1, stride=4, latent_dim=128, save_dir = "/scratch_net/biwidl304/amarugg/gluTracker/weights"):
        super().__init__()
        self.fnet = BasicEncoder(stride=stride)
        print(f"Feature Net consisting of {sum(p.numel() for p in self.fnet.parameters())} Parameters")
        self.r_win = r_win
        self.win_size = 2 * r_win + 1
        #self.predictor = PredictionMLP(self.win_size*self.win_size)
        print(f"Predictor consiting of {sum(p.numel() for p in self.predictor.parameters())} Parameters")
        self.stride = stride
        self.n_corr_lvl = n_corr_lvl
        self.latent_dim = latent_dim
        self.safe_dir = save_dir


    def save(self):
        torch.save(self.state_dict(), self.safe_dir + "/gluTracker.pth")

    def load(self):
        self.load_state_dict(torch.load(self.safe_dir + "/gluTracker.pth", map_location='cpu', weights_only=True))

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
    def get_heatmap(self, position, frames):
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
        frames[0] = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        frames[1] = cv2.cvtColor(frames[1], cv2.COLOR_RGB2BGR)
        imgs = np.concatenate((frames[0], frames[1], heatmap), axis=1)
        return imgs


    def sample_features(self, fmap, points):
        C, H, W = fmap.shape
        N, _ = points.shape

        device = points.device

        lin_offset = torch.linspace(-self.r_win, self.r_win, steps=self.win_size, device=device)
        dy, dx = torch.meshgrid(lin_offset, lin_offset, indexing="ij")
        offset_grid = torch.stack((dx, dy), dim=-1).view(-1, 2) # (2*r+1)^2, 2

        #Normalise to [-1; 1]
        offset_grid[:, 0] = offset_grid[:, 0] / (W - 1) * 2
        offset_grid[:, 1] = offset_grid[:, 1] / (H - 1) * 2

        points = points * 2 - 1

        grid = points[:,None,:] + offset_grid[None,:,:]
        grid = grid.view(N, self.win_size, self.win_size, 2) 

        fmap = fmap[None,:,:,:].expand((N, -1, -1, -1))

        sampled = F.grid_sample(fmap, grid, align_corners=True, mode="bilinear", padding_mode="zeros") 
        
        return sampled
    
    """
    Return Features at each queriy location for all correlation levels
    [n_corr, N, w^2]
    """
    def get_anchor_features(self, fmaps_pyramid, n_queries, s):
        device = fmaps_pyramid[0].device
        anchor_features = torch.zeros(self.n_corr_lvl, n_queries.shape[0], self.latent_dim, self.win_size, self.win_size, device=device)
        for i in range(self.n_corr_lvl):
            fmap = fmaps_pyramid[i][s,:,:,:]
            points = self.sample_features(fmap, n_queries)
            anchor_features[i,:,:,:,:] = points
        return anchor_features

    def resize_feature_map(self, fmap, target_size=(108,135)):
        resized = F.interpolate(fmap, size=target_size, mode="bilinear", align_corners=False)
        return resized

    def extract_fmaps_pyramids(self, frames):
        S, C, H, W = frames.shape
        dtype = frames.dtype

        #Normalise video
        frames = 2 * (frames / 255) - 1 # S, 3, 384, 512

        fmap = self.fnet(frames) # N, 128, 96, 128
        fmap = fmap.to(dtype)

        #Rescale fmap to have h/w that is multiple of win size
        fmap = self.resize_feature_map(fmap)

        #Build feature Pyramids
        fmaps_pyramid = []
        fmaps_pyramid.append(fmap)
        for i in range(self.n_corr_lvl - 1):
            fmap = F.avg_pool2d(fmap, kernel_size=self.win_size, stride=self.win_size)
            fmaps_pyramid.append(fmap)

        return fmaps_pyramid
    
    def create_corr_vol(self, features_o, features_n):
        #From LocoTrack
        """
        rename to query features and tracking features
        corr_4D
        """
        N, C, H, W = features_o.shape

        features_old_flat = features_o.view(N, C, H * W)
        feature_new_flat = features_n.view(N, C, H * W)

        # Normalize along the channel dimension
        features_old_norm = F.normalize(features_old_flat, p=2, dim=1) 
        feature_new_norm = F.normalize(feature_new_flat, p=2, dim=1)  

        # Compute cosine similarity: (N, H*W, H_*W_)
        corr_vol = torch.einsum('nch, nck -> nhk', features_old_norm, feature_new_norm)

        corr_vol = corr_vol.view(N, H, W, H, W)
        corr_vol = corr_vol.reshape(N, self.win_size**4)

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
        device = frames.device
        self.train()
        loss = 0
        iter = 0
        for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
            fmaps_pyramid = self.extract_fmaps_pyramids(frames[b,:,:,:,:])
            B, S, N, _ = trajs.shape
            S, C, H, W = fmaps_pyramid[0].shape
            anchors = torch.zeros(self.n_corr_lvl, N, C, self.win_size, self.win_size, device=device)
            trajs_pred = torch.zeros(S, N, 2, device=device)
            for j in range(frames.shape[1] - 1):

                qrs_msk = qrs[b,:,0] <= j # N
                new_qrs = qrs[b,:,0] == j
                trajs_pred[j,:,:][new_qrs] = trajs[b,j,:,:][new_qrs]
                queries = trajs_pred[j,:,:][qrs_msk]

                anchors[:,new_qrs] = self.get_anchor_features(fmaps_pyramid, trajs_pred[j,:,:][new_qrs], j)

                gt_traj = trajs[b, j+1, :, :][qrs_msk]
                d_traj = gt_traj - queries
                gt_vis = visibility[b, j+1, :][qrs_msk]
                
                pred_pos = queries 
                N = queries.shape[0]
                for i in range(self.n_corr_lvl):
                    fmaps = fmaps_pyramid[self.n_corr_lvl - i - 1][j:j+2]
                    _, C,  H, W = fmaps.shape

                    features_last = self.sample_features(fmaps[0,:,:,:], queries)
                    features_cur = self.sample_features(fmaps[1,:,:,:], pred_pos)
                    
                    corr_st = self.create_corr_vol(features_last, features_cur)
                    corr_lt = self.create_corr_vol(anchors[i, qrs_msk], features_cur)

                    corr_vols = torch.cat((corr_st, corr_lt), 1)

                    pred, vis = self.predictor(corr_vols)
                    pred = pred.reshape(N, self.win_size, self.win_size)
                    
                    loss += loss_f(pred, vis, d_traj, gt_vis, H, W)
                    iter += 1

                    d_x, d_y = self.calculate_position_update(pred, H, W)

                    pred_pos[:,0] += d_x
                    pred_pos[:,1] += d_y
        loss.backward()

        return loss.item() / iter
    
    def forward(self, queries: query, frames, n_queries=torch.tensor([])):
        S, C, H, W = frames.shape
        device = frames.device
        fmaps_pyramid = self.extract_fmaps_pyramids(frames)

        #Create New Queries
        if n_queries.shape[0] > 0:
            n_anchors = self.get_anchor_features(fmaps_pyramid, n_queries, 0)
            queries.anchors = torch.cat((queries.anchors, n_anchors), 1)
            queries.coordinates = torch.cat((queries.coordinates, n_queries), 0)

        N, _ = queries.coordinates.shape

        pred_pos = queries.coordinates
        pred_vis = torch.zeros(N, device=device) 
        #Coarse to fine search
        for i in range(self.n_corr_lvl):

            features_last = self.sample_features(fmaps_pyramid[self.n_corr_lvl - i - 1][0,:,:,:], queries.coordinates)
            features_cur = self.sample_features(fmaps_pyramid[self.n_corr_lvl - i - 1][1,:,:,:], pred_pos)

            corr_st = self.create_corr_vol(features_last, features_cur)
            corr_lt = self.create_corr_vol(queries.anchors[i,:,:,:], features_cur)

            corr_vol = torch.cat((corr_st, corr_lt), 1)


            pred, vis = self.predictor(corr_vol)
            pred = pred.reshape(N, self.win_size, self.win_size)
            vis = vis.reshape(N)

            d_x, d_y = self.calculate_position_update(pred, fmaps_pyramid[self.n_corr_lvl - i - 1].shape[2], fmaps_pyramid[self.n_corr_lvl - i - 1].shape[3])

            pred_pos[:,0] += d_x
            pred_pos[:,1] += d_y

            pred_vis[:] += vis
        
        pred_vis /= self.n_corr_lvl

        queries.coordinates = pred_pos.detach()

        return queries, pred_vis.detach()