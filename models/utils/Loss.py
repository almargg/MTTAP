import torch
import torch.nn as nn
import torch.nn.functional as F

def huber_loss(pred, target, delta=6.0):
    abs_diff = torch.norm(pred - target, dim=1)
    flag = (abs_diff < delta).to(torch.int8)
    return flag * 0.5 * abs_diff ** 2 + (1 - flag) * delta*(abs_diff - 0.5 * delta)

def track_loss(tracks_gt: torch.Tensor, vis_gt: torch.Tensor, tracks_pred: torch.Tensor, vis_pred: torch.Tensor, valid: torch.Tensor):
    """
    tracks_gt: Tensor of shape (S, N, 2) with ground truth track positions
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    tracks_pred: Tensor of shape (S, N, 2) with predicted track positions
    vis_pred: Tensor of shape (S, N) with predicted visibility values
    valid: Tensor of shape (S, N) indicating valid samples
    """
    S, N, _ = tracks_gt.shape
    gamma = 0.95
    weight_vis = 1
    weight_invis = 0.2
    use_dicount = True

    weights = torch.pow(gamma, torch.arange(0, S, 1, device=tracks_gt.device))
    
    vs_msk = vis_gt == True

    visible_loss = huber_loss(tracks_pred[vs_msk], tracks_gt[vs_msk]) * weight_vis
    invisible_loss = torch.norm(tracks_pred[~vs_msk] - tracks_gt[~vs_msk], dim=1) * weight_invis

    loss_tensor = torch.zeros(S, N).to(tracks_gt.device)
    loss_tensor[vs_msk] = visible_loss
    loss_tensor[~vs_msk] = invisible_loss
    
    if use_dicount:
        loss_tensor = loss_tensor * weights[:, None]

    return torch.mean(loss_tensor[valid])

def vis_loss(vis_gt: torch.Tensor, vis_pred: torch.Tensor, valid: torch.Tensor):
    """
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    vis_pred: Tensor of shape (S, N) with predicted visibility values
    valid: Tensor of shape (S, N) indicating valid samples
    """
    S, N = vis_gt.shape
    gamma = 0.95
    weights = torch.pow(gamma, torch.arange(0, S-1, 1, device=vis_gt.device))
    use_discount = False

    # Clamp visibility predictions to avoid log(0)
    vis_pred = torch.clamp(vis_pred, min=0, max=1)

    bce = -(vis_gt * torch.log(vis_pred + 1e-8) + (1 - vis_gt) * torch.log(1 - vis_pred + 1e-8))

    if use_discount:
        bce = bce * weights[:, None]

    return torch.mean(bce[valid])

def confidence_loss(tracks_gt: torch.Tensor, vis_gt: torch.Tensor, tracks_pred: torch.Tensor, conf_pred: torch.Tensor, valid: torch.Tensor):
    """
    tracks_gt: Tensor of shape (S, N, 2) with ground truth track positions
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    tracks_pred: Tensor of shape (S, N, 2) with predicted track positions
    conf_pred: Tensor of shape (S, N) with predicted confidence values
    valid: Tensor of shape (S, N) indicating valid samples
    """
    expected_dist_thresh = 12
    err = torch.sum(torch.square(tracks_gt - tracks_pred), dim=2)  # S, N
    within_thresh = (err < expected_dist_thresh**2).to(torch.int8)  # S, N

    #clamp confidence predictions to avoid log(0)
    conf_pred = torch.clamp(conf_pred, min=0, max=1)

    bce = -(within_thresh * torch.log(conf_pred + 1e-8) + (1 - within_thresh) * torch.log(1 - conf_pred + 1e-8))

    #TODO also use invisible samples
    bce = bce * vis_gt
    return torch.mean(bce[valid])

    
def track_loss_with_confidence(tracks_gt: torch.Tensor, vis_gt: torch.Tensor, tracks_pred: torch.Tensor, vis_pred: torch.Tensor, conf_pred: torch.Tensor, valid: torch.Tensor):
    """
    tracks_gt: Tensor of shape (S, N, 2) with ground truth track positions
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    tracks_pred: Tensor of shape (S, N, 2) with predicted track positions
    vis_pred: Tensor of shape (S, N) with predicted visibility values
    conf_pred: Tensor of shape (S, N) with predicted confidence values
    valid: Tensor of shape (S, N) indicating valid samples
    """

    # Print the shape of all inputs
    #print(f"tracks_gt shape: {tracks_gt.shape}, vis_gt shape: {vis_gt.shape}, "
    #      f"tracks_pred shape: {tracks_pred.shape}, vis_pred shape: {vis_pred.shape}, "
    #      f"conf_pred shape: {conf_pred.shape}, valid shape: {valid.shape}", flush=True)    
                                                               
    track_weight = 0.05
    vis_weight = 1.0 # 0.7
    conf_weight = 1.0 # 2.0

    loss_track = track_loss(tracks_gt, vis_gt, tracks_pred, vis_pred, valid)
    loss_vis = vis_loss(vis_gt, vis_pred, valid)
    loss_conf = confidence_loss(tracks_gt, vis_gt, tracks_pred, conf_pred, valid)

    #print(f"Scales: vis: {loss_vis.item()/loss_track.item()}, conf: {loss_conf.item()/loss_track.item()}")

    loss = loss_track * track_weight + loss_vis * vis_weight + loss_conf * conf_weight

    return loss



#In Cotracker additionally a discount is used
def coTrack_loss(pos_gt: torch.Tensor, vis_gt: torch.Tensor, pos_pred: torch.Tensor, vis_pred: torch.Tensor):
    delta = 16.0 / 256.0
    loss = nn.HuberLoss(reduction='none', delta=delta)

    dist = torch.sqrt(torch.sum(torch.square(pos_gt - pos_pred), 2))
    zeros = torch.zeros_like(dist)

    loss_track = loss(target=zeros, input=dist) * ((vis_gt < 0.5)* 0.2 + (vis_gt >= 0.5) )# B, N
    loss_track = torch.mean(loss_track)

    loss = nn.BCELoss()
    loss_vis = loss(vis_pred, vis_gt)

    #Add confidence
    return loss_track + loss_vis

def pixel_pred_loss(pred, vis, d_traj, gt_vis, H, W):
    """
    pred: Tensor of shape (N, l, l) with predicted pixel values
    vis: Tensor of shape (N, 1) with visibility values
    d_traj: Tensor of shape (N, 2) with ground truth trajectory coordinates
    gt_vis: Tensor of shape (N, 1) with ground truth visibility values
    H: Height of the image
    W: Width of the image
    """
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





    