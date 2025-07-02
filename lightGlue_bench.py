import time
import torch
import numpy as np
import cv2
import os
from dataset.Dataloader import TapvidDavisFirst, TapvidRgbStacking, TapData
from utils.Visualise import display_tap_vid, display_torch_frame, save_tap_vid
from utils.Metrics import compute_metrics


from LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from LightGlue.lightglue.utils import load_image, rbd


# match the features
#matches01 = matcher({'image0': feats0, 'image1': feats1})
#feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
#matches = matches01['matches']  # indices with shape (K,2)
#points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
#points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)



def find_closest_point(point, keypoints):
    offset = point - keypoints
    dist = torch.linalg.vector_norm(keypoints - point, ord=2, dim=1)
    idx = torch.argmin(dist)
    d = dist[idx]
    o = offset[idx]
    o = (o[0], o[1])
    return idx, d, o

def preprocess_frame(frame: torch.Tensor):
    return frame / 255.0

def create_video_tracking(frames, qrs, extractor, matcher, device):
    keys = ["keypoints", "scores", "descriptors"]

    frames = frames[0]
    qrs = qrs[0]
    
    S = frames.shape[0]
    N = qrs.shape[0]

    visible = torch.ones(S, N, device=device)
    trajs = torch.zeros(S, N, 2, device=device) - 1
    new_qrs = qrs[:,0] == 0

    #SETUP
    first_frame = preprocess_frame(frames[0,:,:,:]).to(device)
    feats0 = extractor.extract(first_frame)
    keypoints = feats0["keypoints"] # B, N, 2
    scores = feats0["keypoint_scores"] # B, N
    descriptors = feats0["descriptors"] #B, N, 256

    #Find points closest to queries
    trajs[0, :, :] = qrs[:,1:]

    max_dist = 0
    avg_dist = 0
    idxs = []
    offset = []
    for j in range(qrs.shape[0]):
        idx, dist, off = find_closest_point(qrs[j,1:].to(device), keypoints[0])
        idxs.append(idx)
        offset.append(off)
        if dist>max_dist:
            max_dist = dist
        avg_dist += dist / qrs.shape[0]

    #print(f"Largest distance to trackable point: {max_dist}")
    #print(f"Average distance to trackable point: {avg_dist}")
    ofsts = torch.tensor(offset, device=device)
    retain_ind = torch.tensor(idxs)
    #feats0["keypoints"] = keypoints[:,retain_ind,:]
    #feats0["keypoint_scores"] = scores[:,retain_ind]
    #feats0["descriptors"] = descriptors[:,retain_ind,:]

    for i in range(S - 1):
        new_frame = preprocess_frame(frames[i + 1]).to(device)
        feats1 = extractor.extract(new_frame)
        matches01 = matcher({'image0': feats0, 'image1': feats1})

        feats00, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] # remove batch dimension  # indices with shape (K,2)
        matches = matches01['matches']
        points0 = feats00['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]
        #mat = matches[0,:]  # coordinates in image #1, shape (K,2)
        
        ret_idx = torch.zeros_like(retain_ind, device=device)
        valid = torch.zeros_like(retain_ind, device=device) == 1
        for j in range(retain_ind.shape[0]):
            idx = (matches[:,0] == retain_ind[j]).nonzero(as_tuple=False)
            if idx.shape[0] > 0:
                ret_idx[j] = idx
                valid[j] = True

        trajs[i, valid] = points1[ret_idx[valid]] + ofsts[valid]

        
        #trajs[i+1,:,:][valid] = pts
        #display_torch_frame(frames[i+1], pts)

    for j in range(qrs.shape[0]):
        last_point = qrs[j,1:]
        for i in range(S):
            if trajs[i,j,0] == -1:
                trajs[i,j,:] = last_point
            else:
                last_point = trajs[i,j,:]

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


#extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
extractor = DISK(max_num_keypoints=2048).eval().to(device)
#extractor = ALIKED(max_num_keypoints=2048).eval().to(device)



#matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher
matcher = LightGlue(features='disk').eval().to(device)  # load the matcher
#matcher = LightGlue(features='aliked').eval().to(device)  # load the matcher


#image0 = load_image('C:/Users/alexm/Documents/Unterlagen_ETH/Master/Sem4/MT/Videos/frames/frame_0000.png').to(device)

with torch.no_grad():
    #features = extractor.extract(image0)
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
            
            pred_tracks, pred_visibility = create_video_tracking(frames, qrs, extractor, matcher, device)

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
            
            if j % 5 == 0:
                name = f"{dataset_names[i]}_seq_{j}.gif"
                path = os.path.join("/scratch_net/biwidl304/amarugg/files/videos", name)
                save_tap_vid(data, path)

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

