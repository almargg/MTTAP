import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Modules import BasicEncoder, EfficientUpdateFormer, Mlp
import cv2
import numpy as np
from dataclasses import dataclass
from models.utils.util import posenc, get_1d_sincos_pos_embed_from_grid, bilinear_sampler
import time

@dataclass
class query():
    coordinates: torch.Tensor
    anchors: torch.Tensor

"""
Tracker of any point utilizing depth information to track the point in a video sequence.
"""    
class CoTrackerfunctions():
    def __init__(self):
        self.corr_levels = 4
        self.corr_radius = 3
        self.stride = 4
        self.latent_dim = 128
        pass
    
    def bilinear_sampler(self, input, coords, align_corners=True, padding_mode="border"):
        """Sample a tensor using bilinear interpolation

        `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
        coordinates :attr:`coords` using bilinear interpolation. It is the same
        as `torch.nn.functional.grid_sample()` but with a different coordinate
        convention.

        The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
        :math:`B` is the batch size, :math:`C` is the number of channels,
        :math:`H` is the height of the image, and :math:`W` is the width of the
        image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
        interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

        Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
        in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
        that in this case the order of the components is slightly different
        from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

        If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
        in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
        left-most image pixel :math:`W-1` to the center of the right-most
        pixel.

        If `align_corners` is `False`, the coordinate :math:`x` is assumed to
        be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
        the left-most pixel :math:`W` to the right edge of the right-most
        pixel.

        Similar conventions apply to the :math:`y` for the range
        :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
        :math:`[0,T-1]` and :math:`[0,T]`.

        Args:
            input (Tensor): batch of input images.
            coords (Tensor): batch of coordinates.
            align_corners (bool, optional): Coordinate convention. Defaults to `True`.
            padding_mode (str, optional): Padding mode. Defaults to `"border"`.

        Returns:
            Tensor: sampled points.
        """

        sizes = input.shape[2:]

        assert len(sizes) in [2, 3]

        if len(sizes) == 3:
            # t x y -> x y t to match dimensions T H W in grid_sample
            coords = coords[..., [1, 2, 0]]

        if align_corners:
            coords = coords * torch.tensor(
                [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
            )
        else:
            coords = coords * torch.tensor(
                [2 / size for size in reversed(sizes)], device=coords.device
            )

        coords -= 1

        return F.grid_sample(
            input, coords, align_corners=align_corners, padding_mode=padding_mode
        )
    
    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)

        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)

        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):

        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = self.sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def sample_features5d(self, input, coords):
        """Sample spatio-temporal features

        `sample_features5d(input, coords)` works in the same way as
        :func:`sample_features4d` but for spatio-temporal features and points:
        :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
        a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
        x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

        Args:
            input (Tensor): spatio-temporal features.
            coords (Tensor): spatio-temporal points.

        Returns:
            Tensor: sampled features.
        """

        B, T, _, _, _ = input.shape

        # B T C H W -> B C T H W
        input = input.permute(0, 2, 1, 3, 4)

        # B R1 R2 3 -> B R1 R2 1 3
        coords = coords.unsqueeze(3)

        # B C R1 R2 1
        feats = self.bilinear_sampler(input, coords)

        return feats.permute(0, 2, 3, 1, 4).view(
            B, feats.shape[2], feats.shape[3], feats.shape[1]
        )  # B C R1 R2 1 -> B R1 R2 C
    
    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = self.bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def create_corr_volume(self, fmaps_pyramid, track_feat_support_pyramid, coords):
        """
        Create correlation volumes for the given coordinates and feature maps.
        Args:
            fmaps_pyramid: List of feature maps at different scales. each of shape (C, H, W)
            track_feat_support_pyramid: List of track features at different scales. each of shape (B, T, S, S, N, C)??
            coords: Tensor of coordinates with shape (B, T, N, 2) where T is the number of frames and N is the number of queries.
        """
        B, S, N, _ = coords.shape
        coords_init = coords.view(B * S, N, 2)
        r = 2 * self.corr_radius + 1
        corr_vols = []
        for i in range(self.corr_levels):
            corr_feat = self.get_correlation_feat(
                fmaps_pyramid[i],
                coords_init / 2**i
            )
            track_feat_support = (
                track_feat_support_pyramid[i]
                .view(B, 1, r, r, N, self.latent_dim)
                .squeeze(1)
                .permute(0, 3, 1, 2, 4)
            )
            corr_vol = torch.einsum(
                "btnhwc, bnijc->btnhwij", corr_feat, track_feat_support
            )
            corr_vols.append(corr_vol)

        return corr_vols

    def create_track_feat_pyramid(self, fmaps_pyramid, queries):
        """
        Create a pyramid of track features for the given queries.
        Args:
            fmaps_pyramid: List of feature maps at different scales. each of shape (C, H, W)
            queries: Tensor of queries with shape (N, 2) where N is the number of queries.
        """
        queried_frames = torch.zeros(1,queries.shape[0], device=queries.device)
        queried_coords = queries[None]  # (B, N, 2)
        track_feat_support_pyramid = []
        for i in range(self.corr_levels):
            tracks_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )
            track_feat_support_pyramid.append(track_feat_support)
        return track_feat_support_pyramid

class DepthTracker(nn.Module):
    def __init__(
        self,
        win_len = 16,
        corr_radius = 3,
        corr_levels = 4,
        corr_stride = 4,
        model_resolution = (384, 512),
        num_virtul_tracks = 64,

    ):
        super().__init__()
        self.cotracker_functions = CoTrackerfunctions()
        self.win_len = win_len
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.corr_stride = corr_stride
        self.model_resolution = model_resolution
        self.init = None
        self.tracks = None

        mlp_out_dim = 256
        self.latend_dim = 128

        self.pos_min_degree = 0
        self.pos_max_degree = 10
        self.pos_encodeing_dim = 84

        self.transformer_in = corr_levels * mlp_out_dim + self.pos_encodeing_dim + 2 # 2 for  visibility and confidence

        # TODO: Change to learnable time embedding
        time_grid = torch.linspace(0, win_len - 1, win_len).reshape(win_len, 1)
        self.time_embedding = get_1d_sincos_pos_embed_from_grid(self.transformer_in, time_grid)

        self.fnet = BasicEncoder(stride=corr_stride, output_dim=self.latend_dim)

        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.transformer_in,
            hidden_size=384,
            num_heads=8,
            output_dim=4,
            mlp_ratio=4,
            num_virtual_tracks=num_virtul_tracks,
            add_space_attn=True,
            linear_layer_for_vis_conf=True,
        )

        self.corr_mlp = Mlp(
            in_features= (self.corr_radius* 2 + 1)**4,
            hidden_features= 384,
            out_features= mlp_out_dim,
        )

    def save(self, safe_dir = "/scratch_net/biwidl304/amarugg/gluTracker/weights"):
        torch.save(self.state_dict(), safe_dir + "/depth_Tracker.pth")

    def load_individual_weights(self, safe_dir="/scratch_net/biwidl304/amarugg/gluTracker/weights"):
        self.fnet.load_state_dict(torch.load(safe_dir + "/fnet.pth", map_location='cpu', weights_only=True))
        self.corr_mlp.load_state_dict(torch.load(safe_dir + "/corr_mlp.pth", map_location='cpu', weights_only=True))
        self.updateformer.load_state_dict(torch.load(safe_dir + "/updateformer.pth", map_location='cpu', weights_only=True))

    def load(self, safe_dir= "/scratch_net/biwidl304/amarugg/gluTracker/weights"):
        self.load_state_dict(torch.load(safe_dir + "/depth_Tracker.pth", map_location='cpu', weights_only=True))

    def getsupport_points(self, coordinates):
        """
        Create grid of sample_win^2 support points around the given coordinates.
        Args:
            coordinates: Normalised coordinates in [0, 1] of shape (N, 2)
        Returns:
            corr_features: Correlation features of shape (sample_win^2, N, 3)
        """
        N = coordinates.shape[0]
        device = coordinates.device

        # Convert coordinates to pixel space
        centroid_lvl = coordinates

        # Create a grid for sampling
        dx = torch.linspace(-self.corr_radius, self.corr_radius, 2 * self.corr_radius + 1, device=device)
        dy = torch.linspace(-self.corr_radius, self.corr_radius, 2 * self.corr_radius + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dx, dy, indexing='ij')
        #zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([xgrid, ygrid], axis=-1)
        delta_lvl = delta.view(2 * self.corr_radius + 1, 2 * self.corr_radius + 1, 2)
        
        coords_lvl = centroid_lvl + delta_lvl.view(-1, 2).unsqueeze(0)  # (N, (2 * corr_radius + 1)^2, 2)

        return coords_lvl

    def get_corr_features(self, fmap, coordinates):
        """
        Args:   
            fmap: Feature map of shape (C, H, W)
            coordinates: Normalised coordinates in [0, 1] of shape (N, 2)
            Returns:
                corr_features: Correlation features of shape (N, S, S, C) where S = 2 * self.corr_radius + 1
        """

        C, H, W = fmap.shape
        N = coordinates.shape[0]
        device = coordinates.device

        sample_coords = coordinates[:, None, :]  # (N, 1, 2)
        support_points = self.getsupport_points(sample_coords)  

        corr_feat = bilinear_sampler(fmap, support_points)
        return corr_feat
        
    def get_features_pyramid(self, frame):

        C, H, W = frame.shape
        dtype = frame.dtype

        #Normalise frame
        frame = 2.0 * (frame / 255.0) - 1.0

        #Extract features
        fmap = self.fnet(frame[None,:,:,:])[0]  # Add batch dimension

        # L2 normalization of feature map
        fmap = fmap / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmap), axis=0, keepdims=True),
                torch.tensor(1e-12, device=fmap.device),
            )
        )

        scale = torch.sum(torch.square(fmap[:,0,0]))

        fmap = fmap.to(dtype)

        fmaps_pyramid = []
        fmaps_pyramid.append(fmap)
        for i in range(self.corr_levels - 1):
            fmap = F.avg_pool2d(fmap, kernel_size=2, stride=2)
            fmaps_pyramid.append(fmap)
        
        return fmaps_pyramid
    
    def get_corr_embs(self, feature_pyramid, coords):

        N = coords.shape[0]
        corr_embeddings = []
        for i in range(self.corr_levels):

            corr_features = self.get_corr_features(feature_pyramid[i], coords / 2**i / self.corr_stride)  # (N, S, S, C)

            corr_volume = torch.einsum('nchw, ncij -> nhwij',corr_features, self.track_features[i])

            corr_embedding = self.corr_mlp(corr_volume.reshape(N, -1))  # (N, C)
            corr_embeddings.append(corr_embedding)




        return torch.stack(corr_embeddings, dim=0).permute(1, 0, 2).reshape(N, -1)#, timings

    
    def add_tracks(self, queries):
        """
        Add new tracks to the tracker.
        Args:
            queries: containing coordinates
        """
        if self.init is None:
            raise ValueError("Tracker not initialized. Call init_tracker first.")
        # Extract features from the last frame
        N = queries.shape[0]
        coor_feature_pyramid = []
        for i in range(self.corr_levels):
            corr_features = self.get_corr_features(self.last_pyr[i], queries / 2**i / self.corr_stride)  # Normalise coordinates to the feature map size
            coor_feature_pyramid.append(corr_features)
        self.track_features = torch.cat((self.track_features, torch.stack(coor_feature_pyramid, dim=0)), dim=1)  # (n_corr_levels, N, S, S, C)

        for i in range(self.corr_levels):
                    self.last_pyr[i] = self.last_pyr[i][None,None,:,:,:]
        track_feat_support_pyramid = self.cotracker_functions.create_track_feat_pyramid(self.last_pyr, queries)
        corr_volumes = self.cotracker_functions.create_corr_volume(self.last_pyr, track_feat_support_pyramid, queries[None, None, :, :])  
        for i in range(self.corr_levels):
            self.last_pyr[i] = self.last_pyr[i][0,0,:,:,:]

        corr_embeddings = []
        for i in range(self.corr_levels):
            corr_emb = self.corr_mlp(corr_volumes[i].reshape(N, -1))  # (N, C)
            corr_embeddings.append(corr_emb)  # (n_corr_levels, N, C)

        visibility = torch.ones((N, 1), device=queries.device).float()  
        confidence = torch.ones((N, 1), device=queries.device).float()  # confidence TODO: change to learnable confidence
        corr_embeddings = torch.stack(corr_embeddings, dim=0).permute(1, 0, 2).reshape(N, -1)  

        self.transformer_input = torch.cat((self.transformer_input, torch.cat((visibility, confidence, corr_embeddings), dim=1).repeat(self.win_len, 1, 1).permute(1, 0, 2)), dim=0)
        self.last_coords = torch.cat((self.last_coords, queries), dim=0)  
        self.tracks = torch.cat((self.tracks, queries.repeat(self.win_len, 1).reshape(-1, self.win_len, 2)), dim=0) 
        self.last_vis = torch.cat((self.last_vis, visibility[:,0]), dim=0)
        self.last_confidence = torch.cat((self.last_confidence, confidence[:,0]), dim=0)
        for i in range(self.corr_levels):
            self.track_feat_support_pyramid[i] = torch.cat((self.track_feat_support_pyramid[i], track_feat_support_pyramid[i]), dim=2)
        

    def init_tracker(self, frame, queries):
        self.init = True

        N = queries.shape[0]

        # Extract features from the first frame
        fmap_pyramid = self.get_features_pyramid(frame)
        self.last_pyr = fmap_pyramid
        coor_feature_pyramid = []
        for i in range(self.corr_levels):
            corr_features = self.get_corr_features(fmap_pyramid[i], queries / 2**i / self.corr_stride)  
            coor_feature_pyramid.append(corr_features)
        self.track_features = torch.stack(coor_feature_pyramid, dim=0)  


        for i in range(self.corr_levels):
                    fmap_pyramid[i] = fmap_pyramid[i][None,None,:,:,:]
        self.track_feat_support_pyramid = self.cotracker_functions.create_track_feat_pyramid(fmap_pyramid, queries)
        corr_volumes = self.cotracker_functions.create_corr_volume(fmap_pyramid, self.track_feat_support_pyramid, queries[None, None, :, :])  
        for i in range(self.corr_levels):
            fmap_pyramid[i] = fmap_pyramid[i][0,0,:,:,:]

        corr_embeddings = []
        for i in range(self.corr_levels):
            corr_emb = self.corr_mlp(corr_volumes[i].reshape(N, -1))  # (N, C)
            corr_embeddings.append(corr_emb)  # (n_corr_levels, N, C)

        # Create transformer inputs
        #corr_embeddings = self.get_corr_embs(fmap_pyramid, queries)
        corr_embeddings = torch.stack(corr_embeddings, dim=0).permute(1, 0, 2).reshape(N, -1)
        visibility = torch.ones((N, 1), device=queries.device).float()  
        confidence = torch.ones((N, 1), device=queries.device).float()  # confidence TODO: change to learnable confidence

        self.transformer_input = torch.cat((visibility, confidence, corr_embeddings), dim=1).repeat(self.win_len, 1, 1).permute(1, 0, 2)  # (N, win_len, transformer_in)
        self.last_coords = queries
        self.tracks = queries.repeat(self.win_len, 1).reshape(-1, self.win_len, 2) 
        self.last_vis = visibility[:,0]
        self.last_confidence = confidence[:,0]

    def reset_tracker(self):
        """
        Reset the tracker.
        """
        self.init = None
        self.track_features = None
        self.last_pyr = None
        self.last_coords = None
        self.tracks = None
        self.transformer_input = None
        

    def forward(self, new_frame, iters=2):
        """
        Args:
            new_frame: (C, H, W)
            new_queries: (nN, 2) - Normalised coordinates in [0, 1]
            iters: Number of iterations to run the updateformer
        Returns:
            tracks: (N, 2) - Normalised coordinates in [0, 1]
        """
        if self.init is None:
            raise ValueError("Tracker not initialized. Call init_tracker first.")
        C, H, W = new_frame.shape
        device = new_frame.device
        dtype = new_frame.dtype

        feature_pyramid = self.get_features_pyramid(new_frame)
        N = self.track_features.shape[1]

        vis = self.last_vis
        confidence = self.last_confidence
        coords = self.last_coords.float()

        self.tracks[:, :-1] = self.tracks[:, 1:].clone()
        self.transformer_input[:,:-1, :] = self.transformer_input[:, 1:, :].clone()
            
        for it in range(iters):
            self.tracks[:, -1] = coords

            #TODO: Add corr vol calculation from cotracker
            for i in range(self.corr_levels):
                    feature_pyramid[i] = feature_pyramid[i][None,None,:,:,:]
            corr_volumes = self.cotracker_functions.create_corr_volume(feature_pyramid, self.track_feat_support_pyramid, coords[None, None, :, :])
            for i in range(self.corr_levels):
                feature_pyramid[i] = feature_pyramid[i][0,0,:,:,:]

            corr_embedings = []
            for i in range(self.corr_levels):
                corr_emb = self.corr_mlp(corr_volumes[i].reshape(N, -1))  # (N, C)
                corr_embedings.append(corr_emb)  # (n_corr_levels, N, C)
                #corr_embedings = self.get_corr_embs(feature_pyramid, coords)
            corr_embedings = torch.stack(corr_embedings, dim=0).permute(1, 0, 2).reshape(N, -1)

            self.transformer_input[:,-1, :] = torch.cat((vis[:,None], confidence[:,None], corr_embedings), dim=1)

            #TODO: Change to single frame difference
            rel_coords_forward = self.tracks[:, :-1] - self.tracks[:, 1:]
            rel_coords_backward = self.tracks[:, 1:] - self.tracks[:, :-1]

            scale = torch.tensor([self.model_resolution[1], self.model_resolution[0]], device=device) / self.corr_stride

            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_coords_forward = torch.nn.functional.pad(rel_coords_forward, (0, 0, 0, 1, 0, 0))
            rel_coords_backward = torch.nn.functional.pad(rel_coords_backward, (0, 0, 1, 0, 0, 0))

            rel_pos_emb = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                self.pos_min_degree,
                self.pos_max_degree
            )

            transformer_input = torch.cat((self.transformer_input,rel_pos_emb), dim=2)

            transformer_input = transformer_input + self.time_embedding[None, :, :].to(device)  # Add time embedding 

            delta = self.updateformer(transformer_input)
            delta = delta[0]

            delta_coords = delta[:, -1, :2]
            delta_vis = delta[:, -1,  2]
            delta_confidence = delta[:, -1, 3]

            vis += delta_vis
            confidence += delta_confidence
            coords += delta_coords

        # Update transformer input
        corr_embedings = self.get_corr_embs(feature_pyramid, coords)
        self.transformer_input[:,-1, :] = torch.cat((vis[:,None], confidence[:,None], corr_embedings), dim=1)

        # Update tracks
        self.last_pyr = feature_pyramid
        self.last_coords = coords
        self.last_vis = vis
        self.last_confidence = confidence
        

        return coords, vis, confidence

        