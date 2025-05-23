import time
import torch
import numpy as np
import cv2
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, TapData
from utils.Visualise import display_tap_vid
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

    N = qrs.shape[1]
    visible = torch.ones(N)
    trajs = torch.zeros(N,2)
    new_qrs = qrs[:,0] == 0

    #SETUP
    first_frame = preprocess_frame(frames[0,:,:,:]).to(device)
    last_data = matching.superpoint({"image": first_frame})
    #Find points closest to queries
    idxs = []
    for j in range(qrs[new_qrs].shape[0]):
        idx, dist = find_closest_point(qrs[new_qrs][j,1:], last_data["keypoints"][0])
        idxs.append(idx)
    retain_ind = torch.tensor(idxs)
    last_data = {
        "keypoints": [last_data["keypoints"][0][retain_ind]],
        "scores": (list(last_data["scores"])[0][retain_ind],),
        "descriptors": [last_data["descriptors"][0][:,retain_ind]],
    }
    last_data = {k+"0": last_data[k] for k in keys}
    last_data["image0"] = first_frame

    for i in range(frames.shape[1] - 1):
        new_frame = preprocess_frame(frames[i + 1]).to(device)
        pred = matching({**last_data, 'image1': new_frame})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].detach().numpy()

        valid = matches >= 0
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        #Setup New Anchor
        last_data = {k+"0": pred[k+"1"] for k in keys}
        last_data['image0'] = new_frame

        


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
        "nms_radius": 1,
        "keypoint_threshold": 0.0005,
        "max_keypoints": -1
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 20,
        "match_threshold": 0.2
    }
    
}

matching = Matching(config).eval().to(device)
keys = ["keypoints", "scores", "descriptors"]


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
        
        pred_tracks, pred_occ = create_video_tracking(frames, qrs, matching, device)

        #Prediction by model

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
    
