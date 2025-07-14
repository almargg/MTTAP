import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, KubricDataset
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'


metric = True

if metric == None:
    from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
else:
    from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    max_depth = 20
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'/scratch_net/biwidl304/amarugg/gluTracker/DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

davis_dat = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")
davis = torch.utils.data.DataLoader(davis_dat, batch_size=1, shuffle=False)

#rgb_stack_dat = TapvidRgbStacking("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl")
#rgb_stack = torch.utils.data.DataLoader(rgb_stack_dat, batch_size=1, shuffle=False)

train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=4)

#raw_img = cv2.imread('your/image/path')
#depth = model.infer_image(raw_img) # HxW raw depth map in numpy
def display_depth_maps(train_loader, model: DepthAnythingV2):
    for i, (frames, trajs, vsbls, qrs) in enumerate(train_loader):
            
            # Move frames to CPU and convert to numpy in one go
            device = frames.device
            dtype = frames.dtype
            frames = frames[0]
            B, C, H, W = frames.shape

            # Convert tensor to numpy and batch convert to BGR
            frames = frames.permute(0, 2, 3, 1).cpu().to(torch.float32).numpy()  # (B, H, W, C)
            frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

            # Batch inference if supported, else fallback to per-frame
            depth_maps = []
            
            input_size=518
            torch_frames = []
            for frame in frames_bgr:
                image, (h, w) = model.image2tensor(frame, input_size)
                torch_frames.append(image)
            tmp = torch.stack(torch_frames, dim=0)[:, 0, :, :, :]  # (B, C, H, W)
            images = torch.from_numpy(np.stack(torch_frames, axis=0)[:, 0, :, :, :])

            with torch.no_grad():
                out = model(tmp)
                

            # Vectorized upsampling
            depth_maps = F.interpolate(
                out.unsqueeze(1), 
                size=(h,w), 
                mode="bilinear", 
                align_corners=True
            ).squeeze(1).cpu().numpy()

            for j in range(frames.shape[0]):
                frame = frames_bgr[j]
                depth_map = depth_maps[j]
                # Normalize depth map to 0-255 for visualization
                depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255.0
                depth_map = cv2.applyColorMap((depth_map).astype('uint8'), cv2.COLORMAP_JET)
                frame = frame.astype('uint8')
                combined = np.hstack([frame, depth_map])
                cv2.imshow('Frame and Depth Map', combined)
                cv2.waitKey(0)  # Wait for a key press to show the next frame
            cv2.destroyAllWindows()


def compute_depth(model, x):
    """
    Compute depth using the DepthAnythingV2 model.
    Args:
        x (Tensor): Input tensor of shape (B, C, H, W).
    Returns:
         Tensor: Depth map of shape (B, 1, H, W).
    """
    import cv2
    import numpy as np

    device = x.device
    dtype = x.dtype
    B, C, H, W = x.shape

    # Convert tensor to numpy and batch convert to BGR
    frames = x.permute(0, 2, 3, 1).cpu().to(torch.float32).numpy()  # (B, H, W, C)
    frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    # Batch inference if supported, else fallback to per-frame
    depth_maps = []
        
    input_size=518
    frames = []
    for frame in frames_bgr:
        image, (h, w) = model.image2tensor(frame, input_size)
        frames.append(image)
    frames = torch.stack(frames, dim=0)[:, 0, :, :, :].to(device)  # (B, H, W, C)
    with torch.no_grad():
        out = model(frames)
    depth_maps = F.interpolate(
        out.unsqueeze(1), 
        size=(h,w), 
        mode="bilinear", 
        align_corners=True
    ).squeeze(1)
    # Stack and convert to tensor
        
    out_tensor = depth_maps.unsqueeze(1).to(dtype).to(device)  # (B, 1, H, W)
    return out_tensor

def sample_depth(depth_map, tracks):
    """
    Sample depth values from a depth map based on track coordinates.
    
    Args:
        depth_map (torch.Tensor): Depth maps of shape (S, H, W).
        tracks (torch.Tensor): tensor with track coordinates of shape (S, N, 2) where N is the number of tracks.
        depth (torch.Tensor): Depth values from the dataset of shape (S, N).
        
    Returns:
        torch.tensor: Sampled depth values of shape (S, N).
    """
    if not isinstance(depth_map, torch.Tensor):
        raise TypeError("depth_map must be a torch.Tensor")
    if not isinstance(tracks, torch.Tensor):
        raise TypeError("tracks must be a torch.Tensor")
    
    if depth_map.dim() != 3:
        raise ValueError("depth_map must have shape (S, H, W)")
    if tracks.dim() != 3 or tracks.size(2) != 2:
        raise ValueError("tracks must have shape (S, N, 2) where N is the number of tracks")
    
    S, H, W = depth_map.shape
    S_tracks, N, _ = tracks.shape
    
    if S != S_tracks:
        raise ValueError("The first dimension of depth_map and tracks must match (S)")
    
    # Multiply the track coordinates by the width and height of the depth map
    tracks = tracks.clone()  # Avoid modifying the original tensor
    tracks[:, :, 0] = (tracks[:, :, 0] * W).long()  # x-coordinates
    tracks[:, :, 1] = (tracks[:, :, 1] * H).long()  # y-coordinates 

    # Remove all tracks that are out of bounds
    valid_tracks = (tracks[:, :, 0] >= 0) & (tracks[:, :, 0] < W) & (tracks[:, :, 1] >= 0) & (tracks[:, :, 1] < H)
    
    # Clamp the coordinates to ensure they are within bounds
    tracks[:, :, 0] = torch.clamp(tracks[:, :, 0], 0, W - 1)
    tracks[:, :, 1] = torch.clamp(tracks[:, :, 1], 0, H - 1)

    # Cast to long for indexing
    tracks = tracks.long()
    
    sampled_depths = []
    for s in range(S):
        sampled_depths.append(depth_map[s][tracks[s][:, 1], tracks[s][:, 0]])
    
    sampled_depths = torch.stack(sampled_depths, dim=0)
    #depth = depth[valid_tracks]  # Keep only the valid depth maps
    return sampled_depths, valid_tracks  # Shape: (S, N)
    
def plot_depth_points(sampled_depths, dataset_depths, mask):
    """
    Plot sampled depth points against dataset depth values.
    
    Args:
        sampled_depths (torch.Tensor): Sampled depth values of shape (S, N).
        dataset_depths (torch.Tensor): Depth values from the dataset of shape (S, N).
    """
    import matplotlib.pyplot as plt
    
    if not isinstance(sampled_depths, torch.Tensor):
        raise TypeError("sampled_depths must be a torch.Tensor")
    if not isinstance(dataset_depths, torch.Tensor):
        raise TypeError("dataset_depths must be a torch.Tensor")
    
    
    S, N = sampled_depths.shape
    if dataset_depths.shape != (S, N):
        raise ValueError("sampled_depths and dataset_depths must have the same shape")
    
    # Flatten the tensors for plotting
    sampled_depths = sampled_depths[mask]
    dataset_depths = dataset_depths[mask]

    num_displays = 1000

    # randomly sample points for display
    if sampled_depths.shape[0] > num_displays:
        indices = torch.randperm(sampled_depths.shape[0])[:num_displays]
        sampled_depths = sampled_depths[indices]
        dataset_depths = dataset_depths[indices]


    plt.scatter(sampled_depths.cpu().numpy(), dataset_depths.cpu().numpy(), label='Sampled Depth', color='blue', alpha=0.5)
    
    plt.xlabel('Sampled Depth')
    plt.ylabel('Dataset Depth')
    plt.title('Sampled Depth vs Dataset Depth')
    plt.legend()
    plt.grid()
    plt.show()


def sanity_check(train_loader, model):
    for i, (frames, trajs, vsbls, qrs, depth) in enumerate(train_loader):

        frames = frames[0]
        h,w = frames.shape[2], frames.shape[3]
        # Convert tensor to numpy and batch convert to BGR
        frames = frames.permute(0, 2, 3, 1).cpu().to(torch.float32).numpy()  # (B, H, W, C)
        frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        # Batch inference if supported, else fallback to per-frame
        depth_maps = []
            
        input_size=518
        torch_frames = []
        for frame in frames_bgr:
            image, (h, w) = model.image2tensor(frame, input_size)
            torch_frames.append(image)
        tmp = torch.stack(torch_frames, dim=0)[:, 0, :, :, :]  # (B, C, H, W)
        tmp = tmp.to(DEVICE)
        with torch.no_grad():
            outputs = []
            for i in range(tmp.shape[0]):
                out = model(tmp[i][None])
                outputs.append(out)
            out = torch.cat(outputs, dim=0)  # Concatenate outputs along batch dimension
        #with torch.no_grad():
        #    out = model(tmp)
        depth_maps = F.interpolate(
            out.unsqueeze(1), 
            size=(h,w), 
            mode="bilinear", 
            align_corners=True
        ).squeeze(1)
        print(f"Batch {i+1}: Output shape: {depth_maps.shape}, Depth range: {depth_maps.min().item()} - {depth_maps.max().item()}")
        print(f"Batch {i+1}: Depth dataset shape: {depth.shape}, Depth range: {depth.min().item()} - {depth.max().item()}")
        sampled_depths, mask = sample_depth(depth_maps, trajs[0])
        plot_depth_points(sampled_depths, depth[0], mask)

#sanity_check(train_loader, model)

outs = []

for i, (frames, trajs, vsbls, qrs) in enumerate(davis):
    out = compute_depth(model, frames[0].to(DEVICE))
    outs.append(out.detach().cpu())
    #min_depth = min(min_depth, out.detach().min().item())
    #max_depth = max(max_depth, out.detach().max().item())
    if i == 0:
        break

out = torch.cat(outs, dim=0)  # Concatenate outputs along batch dimension

out = torch.log10(out + 1e-6)  # Apply log10 to the output tensor
mean = out.mean().item()
std = out.std().item()

print(f"Mean of log10 Depth: {mean}, Std of log10 Depth: {std}")

out = (out - mean) / std  # Center the output tensor by subtracting the mean

import matplotlib.pyplot as plt

# Flatten the tensor for histogram
out_flat = out.cpu().numpy().flatten()

plt.figure(figsize=(8, 4))
plt.hist(out_flat, bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Normalized log10 Depth Values')
plt.xlabel('Normalized log10 Depth')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

            
