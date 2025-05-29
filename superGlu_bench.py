import time
import torch
import numpy as np
import cv2
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, TapData
from utils.Visualise import display_tap_vid, display_torch_frame
from utils.Metrics import compute_metrics
from SuperGluePretrainedNetwork.models.matching import Matching

def find_closest_point(point, keypoints):
    dist = torch.linalg.vector_norm(keypoints - point, ord=2, dim=1)
    idx = torch.argmin(dist)
    d = dist[idx]
    return idx, d

def preprocess_frame(frame: torch.Tensor):
    frame_np = frame.numpy()
    frame_np = np.moveaxis(frame_np, 0, -1)
    frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
    return torch.from_numpy(frame_gray/255).float()[None,None]

def create_video_tracking(frames, qrs, matching: Matching, device):
    keys = ["keypoints", "scores", "descriptors"]

    frames = frames[0]
    qrs = qrs[0]
    
    S = frames.shape[0]
    N = qrs.shape[0]

    visible = torch.ones(S, N)
    trajs = torch.zeros(S, N, 2)
    new_qrs = qrs[:,0] == 0

    #SETUP
    first_frame = preprocess_frame(frames[0,:,:,:]).to(device)
    last_data = matching.superpoint({"image": first_frame})
    #Find points closest to queries

    trajs[0, :, :] = qrs[:,1:]

    max_dist = 0
    avg_dist = 0
    idxs = []
    for j in range(qrs.shape[0]):
        idx, dist = find_closest_point(qrs[j,1:].to(device), last_data["keypoints"][0])
        idxs.append(idx)
        if dist>max_dist:
            max_dist = dist
        avg_dist += dist / qrs.shape[0]

    print(f"Largest distance to trackable point: {max_dist}")
    print(f"Average distance to trackable point: {avg_dist}")

    retain_ind = torch.tensor(idxs)
    last_data = {
        "keypoints": [last_data["keypoints"][0][retain_ind]],
        "scores": (list(last_data["scores"])[0][retain_ind],),
        "descriptors": [last_data["descriptors"][0][:,retain_ind]],
    }
    last_data = {k+"0": last_data[k] for k in keys}
    last_data["image0"] = first_frame


    #pts = last_data["keypoints0"][0]
    #display_torch_frame(frames[0], pts)

    for i in range(S - 1):
        new_frame = preprocess_frame(frames[i + 1]).to(device)
        pred = matching({**last_data, 'image1': new_frame})
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches >= 0
        pts = pred["keypoints1"][0][matches[valid]].cpu()
        trajs[i+1,:,:][valid] = pts
        #display_torch_frame(frames[i+1], pts)
        """
        #Frame by frame tracking
        new_frame = preprocess_frame(frames[i + 1]).to(device)
        pred = matching({**last_data, 'image1': new_frame})
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches >= 0
        mkpts1 = kpts1[matches[valid]]

        pts = pred["keypoints1"][0][matches[valid]]
        display_torch_frame(frames[i+1], pts)
        #Save Tracking locations

        mask = torch.from_numpy(valid)
        trajs[i+1,:,:][mask] = torch.from_numpy(mkpts1)

        #Prepare to track new frame
        if i == S-1:
            break

        #Port pred to last data

        matches = pred['matches0'][0]
        valid = matches >= 0

        kpts_n = last_data['keypoints0'][0]
        scores_n = last_data['scores0'][0]
        desc_n = last_data['descriptors0'][0]

        kpts_n[valid] = pred['keypoints1'][0][matches[valid]]
        scores_n[valid] = pred['scores1'][0][matches[valid]]
        desc_n[:, valid] = pred['descriptors1'][0][:, matches[valid]]

        
        #Set location of new qrs
        new_qrs = qrs[:,0] == i + 1
        idxs = []
        avg_dist = 0
        for j in range(qrs[new_qrs].shape[0]):
            idx, dist = find_closest_point(qrs[new_qrs][j,1:].to(device), pred["keypoints1"][0])
            idxs.append(idx)
            if dist>max_dist:
                max_dist = dist
            avg_dist += dist / qrs.shape[0]

        new_idx = torch.tensor(idxs, dtype=int)

        kpts_q = pred['keypoints1'][0][new_idx]
        scores_q = pred['scores1'][0][new_idx]
        desc_q = pred['descriptors1'][0][:, new_idx]

        kpts_n[new_qrs] = kpts_q
        scores_n[new_qrs] = scores_q
        desc_n[:, new_qrs] = desc_q

        last_data = {
            "keypoints": [kpts_n],
            "scores": (scores_n,),
            "descriptors": [desc_n],
        }

        last_data = {k+"0": last_data[k] for k in keys}
        last_data["image0"] = new_frame
        """
    return trajs, visible

        


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

config = {
    "superpoint": {
        "nms_radius": 4,
        "keypoint_threshold": 0.05,
        "max_keypoints": -1
    },
    "superglue": {
        "weights": "indoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2
    }
    
}

matching = Matching(config).eval().to(device)
keys = ["keypoints", "scores", "descriptors"]

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
            qrs
            
            pred_tracks, pred_visibility = create_video_tracking(frames, qrs, matching, device)

            #Prediction by model
            tracks = pred_tracks.cpu().numpy()
            pred_occ = pred_visibility.cpu()

            data = TapData(
            video=frames,
            trajectory=torch.from_numpy(tracks)[None,:,:,:],
            visibility=pred_occ[None,:,:],
            query=qrs
            )
            predictions.append(data)
        

        #display_tap_vid(predictions[0])

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

