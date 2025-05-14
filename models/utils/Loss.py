import torch
import torch.nn as nn




def tap_loss(pos_gt: torch.Tensor, vis_gt: torch.Tensor, pos_pred: torch.Tensor, vis_pred: torch.Tensor):
    
    diff_pos = torch.square(pos_gt - pos_pred)
    diff_pos = torch.sum(diff_pos, 2)
    diff_pos = torch.mean(diff_pos)

    diff_vis = torch.abs(vis_gt - vis_pred)
    diff_vis = torch.mean(diff_vis)

    return diff_pos + diff_vis


#In Cotracker additionally a discount is used
def coTrack_loss(pos_gt: torch.Tensor, vis_gt: torch.Tensor, pos_pred: torch.Tensor, vis_pred: torch.Tensor):
    delta = 16.0 / 256.0
    loss = nn.HuberLoss(reduction='none', delta=delta)

    dist = torch.sqrt(torch.sum(torch.square(pos_gt - pos_pred), 2))
    zeros = torch.zeros_like(dist)

    loss_track = loss(target=zeros, input=dist) * ((vis_gt < 0.5)* 0.2 + (vis_gt > 0.5) )# B, N
    loss_track = torch.mean(loss_track)

    loss = nn.BCELoss()
    loss_vis = loss(vis_pred, vis_gt)

    #Add confidence
    return loss_track + loss_vis

def pixel_pred_loss(pred, vis, d_traj, gt_vis, H, W):
    N, _ = d_traj.shape
    l = pred.shape[1]
    r = (l-1)/2
    device = d_traj.device
    gt_center_y = d_traj[:, 1] * (H-1)
    gt_center_x = d_traj[:, 0] * (W-1)
    
    in_search = torch.logical_and(torch.logical_and(gt_center_y < r, gt_center_y > -r), torch.logical_and(gt_center_x < r, gt_center_x > -r))

    gt_pred = torch.zeros(N, l, l).to(device)#H, W
    for h in range(l):
        for w in range(l):
            center_y = h - r 
            center_y = torch.tensor(center_y).repeat(1, N).to(device)
            center_x = w - r
            center_x = torch.tensor(center_x).repeat(1, N).to(device)
            y_overlap = torch.clamp(torch.min(center_y, gt_center_y) - torch.max(center_y, gt_center_y) + 1, min=0).to(device)
            x_overlap = torch.clamp(torch.min(center_x, gt_center_x) - torch.max(center_x, gt_center_x) + 1, min=0).to(device)
            gt_pred[:, h, w] = y_overlap * x_overlap
    #Normalise to 1
    #gt_pred = torch.reshape(gt_pred, (N, l*l))

    loss_track = torch.mean(torch.sum(torch.abs(gt_pred - pred),dim=(1,2)) * in_search)

    loss = nn.BCELoss()
    loss_vis = loss(vis, gt_vis)

    return loss_track + loss_vis





    