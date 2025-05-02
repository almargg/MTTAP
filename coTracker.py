import time
import torch
import cv2
import numpy as np
from dataset.Dataloader import TapvidDavisFirst, TapData
from utils.Visualise import display_tap_vid
from utils.Metrics import compute_metrics

VIDEO_INPUT_RESO = (384, 512) # H, W
VIDEO_INPUT_RESO_CV = (512, 384) # W, H

#cv2 ordering H, W, C
def resize_video(video: torch.Tensor):
    video = video.numpy()
    B, S, C, H, W = video.shape
    resized_video = np.zeros((B, S, VIDEO_INPUT_RESO[0], VIDEO_INPUT_RESO[1], C))
    for i in range(video.shape[0]): #B
        for j in range(video.shape[1]): #S
            frame = video[i,j,:,:,:]
            frame = np.transpose(frame, (1,2,0))
            r_frame  = cv2.resize(frame, VIDEO_INPUT_RESO_CV, interpolation = cv2.INTER_AREA) 
            resized_video[i,j] = r_frame
            #cv2.imshow("Frame", r_frame)
            #cv2.waitKey(-1)
            #cv2.destroyAllWindows()

    resized_video = np.transpose(resized_video, (0, 1, 4, 2, 3))
    out = torch.from_numpy(resized_video).float()
    return out

 



val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("WARNING NO GPU AVAILABLE")

model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
model = model.to(device)

predictions = []
gts = []
#Swap x and y of queries
#Scale queries
for i, (frames, trajs, vsbls, qrs) in enumerate(val_loader):
    video_resolution = frames.shape[3:] #H,W    
    gt = TapData(
        video=frames,
        trajectory=trajs* torch.tensor([video_resolution[1], video_resolution[0]]),
        visibility=vsbls,
        query=qrs
    )
    gts.append(gt)
    video = resize_video(frames)
    video.to(device)
    qrs[:,:,1] *= VIDEO_INPUT_RESO[1]
    qrs[:,:,2] *= VIDEO_INPUT_RESO[0]
    qrs.to(device)
    
    model(video_chunk=video, is_first_step=True, grid_size=0, queries=qrs, add_support_grid=True)
    for ind in range(0, video.shape[1] - model.step, model.step):
        pred_tracks, pred_visibility = model(video_chunk=video[:,ind : ind + model.step * 2], grid_size=0, queries=qrs, add_support_grid=True)

    tracks = pred_tracks.cpu().numpy()
    pred_occ = pred_visibility[0].cpu()
    tracks = (tracks * np.array([video_resolution[1], video_resolution[0]]) / np.array([VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0]]))[0]
    data = TapData(
        video=frames,
        trajectory=torch.from_numpy(tracks)[None,:,:,:],
        visibility=pred_occ[None,:,:],
        query=qrs
    )
    predictions.append(data)
    print("Done with ", i)
    break

#display_tap_vid(predictions[0], save=True)
avg_thrh_acc =  0
avg_occ_acc = 0
avg_jac = 0
samples = len(gts)
for i in range(samples):
    thr, occ, jac = compute_metrics(gts[i], predictions[i])
    avg_thrh_acc += thr
    avg_occ_acc += occ
    avg_jac += jac
    
print("AVG Threshold accuracy: ", avg_thrh_acc / samples)
print("Occlusion accuracy ", avg_occ_acc / samples)
print("AVG Jaccard ", avg_jac / samples)
    
    

