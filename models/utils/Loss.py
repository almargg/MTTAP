import torch
import torch.nn as nn
import torch.nn.functional as F

def huber_loss(pred, target, delta=6.0):
    abs_diff = torch.sqrt(torch.sum(torch.square(pred - target), dim=2))
    flag = (abs_diff < delta).to(torch.int8)
    return flag * 0.5 * abs_diff ** 2 + (1 - flag) * delta*(abs_diff - 0.5 * delta)

def track_loss(tracks_gt: torch.Tensor, vis_gt: torch.Tensor, tracks_pred: torch.Tensor, vis_pred: torch.Tensor, valid: torch.Tensor):
    """
    tracks_gt: Tensor of shape (S, N, 2) with ground truth track positions
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    tracks_pred: Tensor of shape (S, N, 2) with predicted track positions
    vis_pred: Tensor of shape (S, N) with predicted visibility values
    valid: Tensor of shape (S, N) indicating valid samples (the ones that come after the query frame)
    """
    S, N, _ = tracks_gt.shape
    gamma = 0.95
    weight_vis = 1
    weight_invis = 0.1
    use_discount = True
    # tracks_pred and tracks_gt are of shape (S, N, 2)

    weights = torch.pow(gamma, torch.arange(0, S, 1, device=tracks_gt.device))
    
    vs_msk = vis_gt.to(torch.bool)

    loss = huber_loss(tracks_pred, tracks_gt)
    # loss[vs_msk] *= weight_vis
    loss[~vs_msk] *= weight_invis
    

    #visible_loss = huber_loss(tracks_pred[vs_msk], tracks_gt[vs_msk]) * weight_vis
    #invisible_loss = torch.sqrt(torch.sum((tracks_pred[~vs_msk] - tracks_gt[~vs_msk]) ** 2, dim=1)) * weight_invis

    #print(f"Visible Loss: {visible_loss.mean()}, Invisible Loss: {invisible_loss.mean()}", flush=True)

    #loss_tensor = torch.zeros(S, N).to(tracks_gt.device)
    #loss_tensor[vs_msk] = visible_loss
    #loss_tensor[~vs_msk] = invisible_loss
    
    if use_discount:
        loss = loss * weights[:, None]

    return torch.mean(loss[valid])

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

    #Compute average distance between predicted and ground truth visibility
    diff = torch.abs(vis_gt - vis_pred)
    #print(f"Visibility Loss: {diff.mean()}", flush=True)
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

    
def track_loss_with_confidence(tracks_gt: torch.Tensor, vis_gt: torch.Tensor, tracks_pred: torch.Tensor, vis_pred: torch.Tensor, conf_pred: torch.Tensor, qrs: torch.Tensor):
    """
    tracks_gt: Tensor of shape (S, N, 2) with ground truth track positions
    vis_gt: Tensor of shape (S, N) with ground truth visibility values
    tracks_pred: Tensor of shape (S, N, 2) with predicted track positions
    vis_pred: Tensor of shape (S, N) with predicted visibility values
    conf_pred: Tensor of shape (S, N) with predicted confidence values
    qrs: Tensor of shape (N, 3) 
    """

    # Print the shape of all inputs
    #print(f"tracks_gt shape: {tracks_gt.shape}, vis_gt shape: {vis_gt.shape}, "
    #      f"tracks_pred shape: {tracks_pred.shape}, vis_pred shape: {vis_pred.shape}, "
    #      f"conf_pred shape: {conf_pred.shape}, valid shape: {valid.shape}", flush=True)    

    valid = torch.zeros(tracks_gt.shape[0], tracks_gt.shape[1], dtype=torch.bool)
    for j in range(tracks_gt.shape[0]):
        valid[j, :] = qrs[:, 0] <= j
                                                               
    track_weight = 0.1
    vis_weight = 0.25 # 0.7
    conf_weight = 0.25 # 2.0

    loss_track = track_loss(tracks_gt, vis_gt, tracks_pred, vis_pred, valid)
    loss_vis = vis_loss(vis_gt, vis_pred, valid)
    loss_conf = confidence_loss(tracks_gt, vis_gt, tracks_pred, conf_pred, valid)

    #print(f"Track Loss: {loss_track}, Vis Loss: {loss_vis}, Conf Loss: {loss_conf}", flush=True)

    loss = loss_track * track_weight + loss_vis * vis_weight + loss_conf * conf_weight

    return loss








    
