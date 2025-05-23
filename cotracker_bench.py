import time
import torch
import numpy as np
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, TapData
from utils.Visualise import display_tap_vid
from utils.Metrics import compute_metrics

davis_dat = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")
davis = torch.utils.data.DataLoader(davis_dat, batch_size=1, shuffle=False)

rgb_stack_dat = TapvidRgbStacking("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl")
rgb_stack = torch.utils.data.DataLoader(rgb_stack_dat, batch_size=1, shuffle=False)

datasets = [davis, rgb_stack]
dataset_names = ["Davis", "RGB_STACKING"]

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("WARNING NO GPU AVAILABLE")

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
step_size = cotracker.step

with torch.no_grad():
    for i, dataset in enumerate(datasets):
        predictions = []
        gts = []
        for j, (frames, trajs, vsbls, qrs) in enumerate(dataset):

            video_resolution = frames.shape[3:] #H,W    
            gt = TapData(
                video=frames,
                trajectory=trajs* torch.tensor([video_resolution[1], video_resolution[0]]),
                visibility=vsbls,
                query=qrs
            )
            gts.append(gt)

            qrs[:,:,1] *= frames.shape[3]
            qrs[:,:,2] *= frames.shape[4]
            qrs.to(device)
            frames.to(device)

            cotracker(video_chunk=frames, is_first_step=True, grid_size=0, queries=qrs, add_support_grid=True)
            for ind in range(0, frames.shape[1] - cotracker.step, cotracker.step):
                pred_tracks, pred_visibility = cotracker(
                    video_chunk=frames[:, ind : ind + cotracker.step * 2].to(device),
                    grid_size=0,
                    queries=qrs.to(device),
                    add_support_grid=True
                )
            tracks = pred_tracks.cpu().numpy()
            pred_occ = pred_visibility[0].cpu()

            data = TapData(
            video=frames,
            trajectory=torch.from_numpy(tracks)[None,:,:,:],
            visibility=pred_occ[None,:,:],
            query=qrs
            )
            predictions.append(data)
        avg_thrh_acc =  0
        avg_occ_acc = 0
        avg_jac = 0
        samples = len(gts)
        
        for i in range(samples):
            thr, occ, jac = compute_metrics(gts[i], predictions[i])
            avg_thrh_acc += thr
            avg_occ_acc += occ
            avg_jac += jac

        print("Results from " + dataset_names[i])
        print("AVG Threshold accuracy: ", avg_thrh_acc / samples)
        print("Occlusion accuracy ", avg_occ_acc / samples)
        print("AVG Jaccard ", avg_jac / samples)
        print("\n")
        
