import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Modules import BasicEncoder, EfficientUpdateFormer, Mlp
import numpy as np
from dataclasses import dataclass
from models.utils.util import posenc, get_1d_sincos_pos_embed_from_grid, bilinear_sampler, get_points_on_a_grid, sample_features5d


class DepthTracker(nn.Module):
    def __init__(
        self,
        
    ):
        super().__init__()
        self.window_len = 16
        self.cotracker = CoTrackerThreeOnline(
            stride=4, corr_radius=3, window_len=self.window_len
            )


    def forward(self, video, queries):
        B, S, C, H, W = video.shape
        B, N, D = queries.shape
        device = video.device
        step = self.window_len // 2
        self.cotracker.init_video_online_processing()

        tracks = torch.zeros((B, S, N, 2), device=device)
        visibility = torch.zeros((B, S, N), device=device)
        confidence = torch.zeros((B, S, N), device=device)

        for ind in range(0, video.shape[1] - (step * 2) + 1, 1):
            pred_tracks, pred_visibility, pred_confidence = self.cotracker(
                video=video[:,ind : ind + step * 2], queries=queries, iters=2
            )
            if ind == 0:
                tracks[:, :step * 2, :, :] = pred_tracks[:, :, :, :]
                visibility[:, :step * 2, :] = pred_visibility[:, :, :]
                confidence[:, :step * 2, :] = pred_confidence[:, :, :]
            idx = ind + step * 2 - 1
            tracks[:, idx, :, :] = pred_tracks[:, -1, :, :]  
            visibility[:, idx, :] = pred_visibility[:, -1, :]  
            confidence[:, idx, :] = pred_confidence[:, -1, :]

        return tracks, visibility, confidence
    
    def save(self, path="/scratch_net/biwidl304/amarugg/gluTracker/weights/depth_Tracker.pth"):
        torch.save(self.cotracker.state_dict(), path
        )

    def load(self, path="/scratch_net/biwidl304/amarugg/gluTracker/weights/depth_Tracker.pth"):
        self.cotracker.load_state_dict(torch.load(path, map_location=self.device))

            

class DepthTrackerOnline(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.window_len = 16
        self.cotracker = CoTrackerOnlinePredictor(checkpoint="/scratch_net/biwidl304/amarugg/cotracker/ckpt/scaled_online.pth",
            window_len=self.window_len,
        ).to(device)
        self.device = device

    def track_video(self, video, queries):
        B, S, C, H, W = video.shape
        B, N, D = queries.shape

        video = video.to(self.device)
        queries = queries.to(self.device)

        #Pad video with window lenght times first frame
        #padding = video[:, 0:1].repeat(1, self.window_len - 1, 1, 1, 1)
        #video = torch.cat([padding, video], dim=1)

        tracks = torch.zeros((B, S, N, 2), device=self.device)
        visibility = torch.zeros((B, S, N), device=self.device)

        #for n in range(N):
        #    if queries[0, n, 0] > 0:
        #        queries[0, n, 0] += self.window_len - 1

        self.cotracker(video_chunk=video, is_first_step=True, queries=queries)
        for ind in range(0, video.shape[1] - (self.cotracker.step * 2) + 1, 1):
            pred_tracks, pred_visibility = self.cotracker(video_chunk=video[:,ind : ind + self.cotracker.step * 2], queries=queries)
            if ind == 0:
                tracks[:, :self.cotracker.step * 2, :, :] = pred_tracks[:, :, :, :]
                visibility[:, :self.cotracker.step * 2, :] = pred_visibility[:, :, :]
            idx = ind + self.cotracker.step * 2 - 1
            tracks[:, idx, :, :] = pred_tracks[:, -1, :, :]  # Get the last step predictions
            visibility[:, idx, :] = pred_visibility[:, -1, :]  # Get the last step predictions

        
        return tracks, visibility


    def forward(self, video, queries):
        pred_tracks, pred_visibility = self.track_video(video, queries)
        return pred_tracks, pred_visibility
    

        
    

class CoTrackerOnlinePredictor(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        window_len=16,
    ):
        super().__init__()
        self.support_grid_size = 6
        model = CoTrackerThreeOnline(
                stride=4, corr_radius=3, window_len=window_len
            )
        #Load the model from the checkpoint
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
                time_emb = state_dict["time_emb"]
                unused = (16 - window_len) // 2
                time_emb = time_emb[:,unused:16-unused,:]
                state_dict["time_emb"] = time_emb
                if "model" in state_dict:
                    state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        torch.save(model.state_dict(), "/scratch_net/biwidl304/amarugg/gluTracker/weights/depth_Tracker_final.pth")

        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        queries: torch.Tensor = None,
    ):
        B, T, C, H, W = video_chunk.shape
        # Initialize online video processing and save queried points
        # This needs to be done before processing *each new video*
        if is_first_step:
            self.model.init_video_online_processing()
            B, N, D = queries.shape
            self.N = N
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
            
            self.queries = queries
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(
            B, T, 3, self.interp_shape[0], self.interp_shape[1]
        )
        
        tracks, visibilities, confidence = self.model(
            video=video_chunk, queries=self.queries, iters=2
        )
            
        visibilities = visibilities * confidence
        thr = 0.6
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )
    

class CoTrackerThreeBase(nn.Module):
    def __init__(
        self,
        window_len=16,
        stride=4,
        corr_radius=3,
        corr_levels=4,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        add_space_attn=True,
        linear_layer_for_vis_conf=True,
    ):
        super(CoTrackerThreeBase, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_dim = 256
        self.latent_dim = 128

        self.linear_layer_for_vis_conf = linear_layer_for_vis_conf
        self.fnet = BasicEncoder(input_dim=3, output_dim=self.latent_dim, stride=stride)

        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution

        self.input_dim = 1110

        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=num_virtual_tracks,
            add_space_attn=add_space_attn,
            linear_layer_for_vis_conf=linear_layer_for_vis_conf,
        )
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(
            1, window_len, 1
        )

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
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
        support_track_feats = sample_features5d(fmaps, support_points)

        return support_track_feats
        

    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)


class CoTrackerThreeOnline(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTrackerThreeOnline, self).__init__(**args)

    def init_video_online_processing(self):
        self.online_ind = 0
        self.online_track_feat = [None] * self.corr_levels
        self.online_track_support = [None] * self.corr_levels
        self.online_coords_predicted = None
        self.online_vis_predicted = None
        self.online_conf_predicted = None
        self.online_feature_cache = None

    def forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        vis=None,
        conf=None,
        iters=2,
    ):
        B, S, *_ = fmaps_pyramid[0].shape
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * S, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r * r))

                corr_embs.append(corr_emb)

            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])

            transformer_input = [vis, conf, corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, -1)
            )

            x = x + self.interpolate_time_embed(x, S)
            x = x.view(B, N, S, -1)  # (B N) T D -> B N T D

            delta = self.updateformer(x, add_space_attn=True)

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            vis = vis + delta_vis
            conf = conf + delta_conf

            coords = coords + delta_coords
            coord_preds.append(coords[..., :2] * float(self.stride))

            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds
    
    def create_fmaps_pyramid(self, video):
        dtype = video.dtype
        B, T, C, H, W = video.shape
        S = self.window_len

        video_reshaped = video.reshape(-1, C, H, W)
        if self.online_feature_cache is not None:
            # Only compute the last frame's features and update the cache efficiently
            new_feature = self.fnet(video_reshaped[-1:].contiguous())
            fmaps = torch.cat((self.online_feature_cache[1:], new_feature), dim=0)
            self.online_feature_cache = fmaps
        else:
            fmaps = self.fnet(video_reshaped)
            self.online_feature_cache = fmaps
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # We compute track features
        fmaps_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * S, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, S, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)
        return fmaps_pyramid
    
    def add_track_support(self, fmaps_pyramid, queries):
        B, N, _ = queries.shape
        S = self.window_len
        step = 1
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride
        device = fmaps_pyramid[0].device

        sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
        left = 0 if self.online_ind == 0 else self.online_ind + (S-step)
        right = self.online_ind + S
        sample_mask = (sample_frames >= left) & (sample_frames < right)
        track_feat_support_pyramid = []
        for i in range(self.corr_levels):
            track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames - self.online_ind,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )

            if self.online_track_support[i] is None:
                self.online_track_support[i] = torch.zeros_like(
                    track_feat_support, device=device
                )

            self.online_track_support[i] = track_feat_support * sample_mask + self.online_track_support[i] * ~sample_mask
            track_feat_support_pyramid.append(
                self.online_track_support[i].unsqueeze(1)
            )
        attention_mask = (queried_frames < self.online_ind + S).reshape(B, 1, N)  # B S N
        track_feat_support_pyramid=[
                attention_mask[:, None, :, :, None] * tfeat
                for tfeat in track_feat_support_pyramid
            ]
        return track_feat_support_pyramid

    def forward(
        self,
        video,
        queries,
        iters=2,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, C, H, W]): video frames
            queries (FloatTensor[B, N, 3]): queries of the form (frame, x, y)
            iters (int): number of iterations
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - conf_predicted (FloatTensor[B, T, N])
        """

        B, T, C, H, W = video.shape
        S = self.window_len
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0
        assert (
            self.online_ind is not None
        ), "Call model.init_video_online_processing() first."
        assert (S == T), "The video length must be equal to the window length."
        B, N, __ = queries.shape

        # B = batch size
        # T = number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)

        # video = B T C H W
        # queries = B N 3
        # coords_init = B T N 2
        # vis_init = B T N 1

        step = 1 
        video = 2 * (video / 255.0) - 1.0

        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride
        
        fmaps_pyramid = self.create_fmaps_pyramid(video)    
        
        track_feat_support_pyramid = self.add_track_support(fmaps_pyramid, queries)
        
        vis_init = torch.zeros((B, S, N, 1), device=device).float()
        conf_init = torch.zeros((B, S, N, 1), device=device).float()
        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()

        if self.online_ind > 0:
            overlap = S - step
            copy_over = (queried_frames < self.online_ind + overlap)[
                :, None, :, None
            ]  # B 1 N 1

            coords_init = torch.where(
                copy_over.expand_as(coords_init), self.coords_prev, coords_init
            )
            vis_init = torch.where(
                copy_over.expand_as(vis_init), self.vis_prev, vis_init
            )
            conf_init = torch.where(
                copy_over.expand_as(conf_init), self.conf_prev, conf_init
            )

        coords, viss, confs = self.forward_window(
            fmaps_pyramid=(
                fmaps_pyramid
            ),
            coords=coords_init,
            track_feat_support_pyramid=track_feat_support_pyramid,
            vis=vis_init,
            conf=conf_init,
            iters=iters,
        )

        coords_predicted = coords[-1][:, :T]
        vis_predicted = viss[-1][:, :T]
        conf_predicted = confs[-1][:, :T]

        coords_prev = coords_predicted[:, step:] / self.stride
        padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)
        self.coords_prev = torch.cat([coords_prev, padding_tensor], dim=1)

        vis_prev = vis_predicted[:, step:, :, None].clone()
        padding_tensor = vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)
        self.vis_prev = torch.cat([vis_prev, padding_tensor], dim=1)

        conf_prev = conf_predicted[:, step:, :, None].clone()
        padding_tensor = conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)
        self.conf_prev = torch.cat([conf_prev, padding_tensor], dim=1)

        self.online_ind += step

        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)

        return coords_predicted, vis_predicted, conf_predicted
