from dataset.Dataloader import TapData
import numpy as np
import torch

#Both shaped B, N, 2
def compute_avg_distance(gt: torch.Tensor, prediction: torch.Tensor):
    gt *= 256
    prediction *= 256
    pxl_dist = torch.mean(torch.norm(gt - prediction, p=2))
    return pxl_dist


def compute_metrics(gt: TapData, prediction: TapData):
    thresholds = np.array([1,2,4,8,16]) #T
    gt_trajectory = gt.trajectory.numpy()
    pred_trajectory = prediction.trajectory.numpy()

    #resize trajectories to image of size 256x256
    gt_trajectory[:,:,:,0] *= 256 / gt.video.shape[4]
    gt_trajectory[:,:,:,1] *= 256 / gt.video.shape[3]
    pred_trajectory[:,:,:,0] *= 256 / prediction.video.shape[4]
    pred_trajectory[:,:,:,1] *= 256 / prediction.video.shape[3]

    query_frame = gt.query.numpy()[...,0] #B, N
    query_frame = np.round(query_frame).astype(np.int32) 
    eye = np.eye(gt_trajectory.shape[1]) #B, S, S
    qry_nmbr_to_frms = np.cumsum(eye, axis=1) - eye 
    eval_points = qry_nmbr_to_frms[query_frame] > 0 #B, N, S
    eval_points = np.transpose(eval_points, (0,2,1)) #B, N, S

    distance = gt_trajectory - pred_trajectory
    norm_dist = np.linalg.norm(distance, axis=3) #B, S, N

    within_thresholds = (norm_dist < thresholds[:, None, None])

    pred_visible = prediction.visibility.numpy() > 0.5
    gt_visible = gt.visibility.numpy() > 0.5
    correct_occlusion = gt_visible == pred_visible #B, S, N

    true_positiv = np.sum(eval_points[None] & within_thresholds & gt_visible[None], axis=(0,2,3))
    false_positiv = np.sum(eval_points[None] & (pred_visible[None] & (~gt_visible[None] | ~within_thresholds)), axis=(0,2,3))
    false_negativ = np.sum(eval_points[None] & (gt_visible[None] & (~pred_visible[None] | ~within_thresholds)), axis=(0,2,3))

    assert true_positiv.shape[0] == 5

    jac = true_positiv / (true_positiv + false_positiv + false_negativ)
    avg_jac = np.average(jac)

    occ_acc = np.sum(correct_occlusion & eval_points, axis=(1)) / np.sum(eval_points, axis=(1))
    avg_occ_acc = np.average(occ_acc)

    n_valid = np.sum(gt_visible & eval_points, axis=(1,2))
    threshold_accuracy = np.sum(within_thresholds & gt_visible[:,None,:,:] & eval_points[:,None,:,:], axis=(2,3)) / n_valid
    avg_thrh_acc = np.average(threshold_accuracy)

    return avg_thrh_acc, avg_occ_acc, avg_jac


    

    
    





    
