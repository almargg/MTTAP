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


    