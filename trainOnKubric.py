import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
from models.Model import GluTracker, query
from models.utils.Loss import tap_loss, coTrack_loss, pixel_pred_loss
from utils.Metrics import compute_metrics, compute_avg_distance
from utils.Visualise import create_tap_vid



def validate(model, loader, writer: SummaryWriter, epoch, device):
    vis_all_n_epoch = 5
    model.eval()

    with torch.no_grad():
        n = 0
        thr_acc, occ, jac = 0, 0, 0

        for i, (frames, trajs, vsbls, qrs) in enumerate(loader):
            frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
            trajs_pred = torch.zeros_like(trajs)
            vis_pred = torch.zeros_like(vsbls)

            for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
                queries = query(
                    coordinates=torch.empty(0).to(device),
                    anchors=torch.empty(0).to(device)
                )
                for j in range(frames.shape[1] - 1):

                    new_ptns = qrs[b,:,0] == j #K
                    n_queries = trajs[b,j,:,:][new_ptns]
                    trajs_pred[b,j,:,:][new_ptns] =  n_queries# Add new tracks

                    qrs_msk = qrs[b,:,0] <= j #K

                    frames_ = torch.stack([frames[b,j,:,:,:], frames[b,j+1,:,:,:]], dim=0)
                    #frames_ = torch.unsqueeze(frames_, 0)

                    queries, pred_vis = model(queries, frames_, n_queries)
                    
                    pred_pose = queries.coordinates
                    trajs_pred[b,j+1,:,:][qrs_msk] = pred_pose[:,:]
                    vis_pred[b,j+1,:][qrs_msk] = pred_vis[:]

            trajs_pred, vis_pred = trajs_pred.cpu(), vis_pred.cpu()
            frames, trajs, vsbls, qrs = frames.cpu(), trajs.cpu(), vsbls.cpu(), qrs.cpu()

            trajs[:,:,:,0] *= frames.shape[4]
            trajs[:,:,:,1] *= frames.shape[3]
            trajs_pred[:,:,:,0] *= frames.shape[4]
            trajs_pred[:,:,:,1] *= frames.shape[3]

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

        return (thr_acc + occ + jac) / n



def train(model: GluTracker, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device):
    steps_per_epoch = 1000
    loss_sum = 0
    iter = 0
    best_perf = 0
    for i, (frames, trajs, vsbls, qrs) in enumerate(train_loader):
        
        #Training
        frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
        optimiser.zero_grad()
        loss_sum += model.train_video(qrs, frames, trajs, vsbls, loss_fnct)
        iter += 1
        optimiser.step()

        #Validation
        if (i % steps_per_epoch) == (steps_per_epoch-1):
            epoch = i // steps_per_epoch
            avg_loss = loss_sum / iter
            writer.add_scalar("Loss/train", avg_loss, epoch)
            avg_loss = 0
            iter = 0

            perf = validate(model, val_loader, writer, epoch, device)
            if perf > best_perf:
                model.save()
                best_perf = perf

        if i == n_steps:
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GluTracker().to(device)

    print(f"Loaded model of {sum(p.numel() for p in model.parameters())} Parameters")

    loss_fnct = pixel_pred_loss

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    model.use_trained_fnet()
    #model.load()
    #optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    validate(model, val_loader, writer, -1, device)
    train(model, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device)

    writer.close()


if __name__ == "__main__":
    main()
