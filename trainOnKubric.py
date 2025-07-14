import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
from models.Model import DepthTracker
from models.utils.Loss import  track_loss_with_confidence
from utils.Metrics import compute_metrics, compute_avg_distance
from utils.Visualise import create_tap_vid


def get_new_queries(qrs, j):
    """
    Get new queries for the current frame.
    :param qrs: Tensor of queries with shape (K, 1)
    :param j: Current frame index
    :return: New queries for the current frame
    """
    new_ptns = qrs[:, 0] == j
    return qrs[new_ptns, 1:], new_ptns


def track_video(model, frames, qrs, device):
    """
    Track a video using the model.
    :param model: The tracking model
    :param frames: Tensor of frames with shape (S, C, H, W)
    :param qrs: Tensor of queries with shape (N, 3)
    :param device: Device to run the model on
    :return: Predicted trajectories, visibility, and confidence scores
    """
    
    S, C, H, W = frames.shape
    N, _ = qrs.shape

    frames, qrs = frames.to(device), qrs.to(device)

    trajs_pred = torch.zeros(S, N, 2, device=device)
    vis_pred = torch.zeros(S, N, device=device)
    confidence_pred = torch.zeros(S, N, device=device)
    valids = torch.zeros(S, N, device=device) == 1

    keys = ["feature_extraction", "data_preparation", "corr_input", "embds",  "transformer"]
    d = {key: 0 for key in keys}

    
    model.reset_tracker() #Reset the model tracker for each batch
    new_ptns, mask = get_new_queries(qrs, 0)
    model.init_tracker(frames[0,:,:,:], new_ptns) #Initialize the model with the first frame and queries
    for j in range(1, S, 1): #Iterate over frames
        coords, vis, confidence, times = model(frames[j,:,:,:]) #Run the model on the current frame
        trajs_pred[j][mask] = coords
        vis_pred[j][mask] = vis
        confidence_pred[j][mask] = confidence

        for key in times:
            if key not in d:
                d[key] = 0
            d[key] += times[key]

        valids[j][mask] = True
        new_ptns, tmp_msk = get_new_queries(qrs, j) #Get new queries for the current frame
        n_new = new_ptns.shape[0]
        if n_new == 0:
            continue
        mask = mask | tmp_msk 
        model.add_tracks(new_ptns)
        
    return trajs_pred, vis_pred, confidence_pred, valids, d


def validate(model, loader, writer: SummaryWriter, epoch, device):
    vis_all_n_epoch = 5
    model.eval()

    with torch.no_grad():
        n = 0
        thr_acc, occ, jac = 0, 0, 0

        for i, (frames, trajs, vsbls, qrs) in enumerate(loader):

            trajs_pred, vis_pred, confidence_pred, _, timings = track_video(model, frames[0], qrs[0], device)
            trajs_pred, vis_pred, confidence_pred = trajs_pred.cpu()[None], vis_pred.cpu()[None], confidence_pred.cpu()[None]


            gt = TapData(
                frames,
                trajs,
                vsbls,
                qrs
            )
            pred = TapData(
                frames,
                trajs_pred,
                vis_pred,
                qrs
            )

            avg_thrh_acc, avg_occ_acc, avg_jac = compute_metrics(gt, pred)
            thr_acc += avg_thrh_acc
            occ += avg_occ_acc
            jac += avg_jac
            n += 1

            if (epoch % vis_all_n_epoch) == 0:
                if (i % 10) == 0:
                    video = create_tap_vid(pred)
                    writer.add_video(f"TapVidDavis_{i}", video[None, :, :, :, :], epoch, fps=10)

        writer.add_scalar("AVGThreshold", thr_acc/n, epoch)
        writer.add_scalar("OCCAccuracy", occ/n, epoch)
        writer.add_scalar("AVGJaccard", jac/n, epoch)

        return jac / n



def train(model, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device):
    steps_per_epoch = 1000
    loss_sum = 0
    iter = 0
    best_perf = 0
    steps = 0

    while True:
        for i, (frames, trajs, vsbls, qrs) in enumerate(train_loader):

            #TODO: Add batch to stabilize training
            model.train()
            optimiser.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            trajs_pred, vis_pred, confidence_pred, valids, timings = track_video(model, frames[0], qrs[0], device)
            loss = loss_fnct(trajs.to(device)[0], vsbls.to(device)[0], trajs_pred, vis_pred, confidence_pred, valids)

            loss.backward()
            loss_sum += loss.item()
            iter += 1
            optimiser.step()

            #Validation
            if (steps % steps_per_epoch) == (steps_per_epoch-1):
                epoch = i // steps_per_epoch
                avg_loss = loss_sum / iter
                writer.add_scalar("Loss/train", avg_loss, epoch)
                loss_sum = 0
                iter = 0

                perf = validate(model, val_loader, writer, epoch, device)
                if perf > best_perf:
                    model.save()
                    best_perf = perf

            steps += 1
            if steps == n_steps:
                return
    

def main():
    lr = 1e-4
    batch_size = 1
    n_steps = 20_001

    now = datetime.now()
    now_str = now.strftime(("%d.%m.%Y_%H:%M:%S"))
    writer_dir = os.path.join("/scratch_net/biwidl304/amarugg/gluTracker/runs", now_str)
    os.mkdir(writer_dir)
    writer = SummaryWriter(log_dir=writer_dir)

    train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
    val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, prefetch_factor=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthTracker().to(device)

    print(f"Loaded model of {sum(p.numel() for p in model.parameters())} Parameters")

    loss_fnct = track_loss_with_confidence

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    validate(model, val_loader, writer, -1, device)
    train(model, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device)

    writer.close()


if __name__ == "__main__":
    main()
