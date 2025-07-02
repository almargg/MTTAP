import time
import torch
import numpy as np
import sys
import os
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, TapData
from utils.Visualise import display_tap_vid, save_tap_vid
from utils.Metrics import compute_metrics
from cotracker.cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor

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


#cotracker = CoTrackerOnlinePredictor(
#                checkpoint="/scratch_net/biwidl304/amarugg/cotracker/ckpt/scaled_online.pth",
#                v2=False,
#                offline=False,
#                window_len=8,
#            ).to(device)

cotracker = CoTrackerPredictor(
                checkpoint="/scratch_net/biwidl304/amarugg/cotracker/ckpt/trained/cotracker_three_final.pth",
                v2=False,
                offline=True,
                window_len=60,
            ).to(device)

cotracker.eval()

#stepsize = cotracker.step

used_time = 0
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

            qrs[:,:,1] *= frames.shape[4]
            qrs[:,:,2] *= frames.shape[3]
            qrs = qrs.to(device)
            frames = frames.to(device)

            torch.cuda.synchronize()
            start = time.time()
            #cotracker(video_chunk=frames, is_first_step=True, grid_size=0, queries=qrs, add_support_grid=True)
            #for ind in range(0, frames.shape[1] - (cotracker.step * 2) + 1, 1):
            #    pred_tracks, pred_visibility = cotracker(video_chunk=frames[:,ind : ind + cotracker.step * 2], grid_size=0, queries=qrs, add_support_grid=True)
            pred_tracks, pred_visibility = cotracker(video=frames, grid_size=0, queries=qrs)
            torch.cuda.synchronize()
            end = time.time()
            used_time += end - start

            tracks = pred_tracks.cpu().numpy()
            pred_occ = pred_visibility[0].cpu()

            data = TapData(
            video=frames.cpu(),
            trajectory=torch.from_numpy(tracks),
            visibility=pred_occ[None,:,:],
            query=qrs.cpu()
            )
            predictions.append(data)

            if j % 5 == 0:
                name = f"{dataset_names[i]}_cot_{j}.gif"
                path = os.path.join("/scratch_net/biwidl304/amarugg/files/videos", name)
                save_tap_vid(data, path)
        avg_thrh_acc =  0
        avg_occ_acc = 0
        avg_jac = 0
        samples = len(gts)
        
        for j in range(samples):
            thr, occ, jac = compute_metrics(gts[j], predictions[j])
            avg_thrh_acc += thr
            avg_occ_acc += occ
            avg_jac += jac

        print("Results from " + dataset_names[i])
        print("AVG Threshold accuracy: ", avg_thrh_acc / samples)
        print("Occlusion accuracy ", avg_occ_acc / samples)
        print("AVG Jaccard ", avg_jac / samples)
        print("\n")

    print("Time taken: ", used_time)
        
