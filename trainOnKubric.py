import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData, seed_everything
from models.Model import DepthTracker
from models.utils.Loss import  track_loss_with_confidence
from utils.Metrics import compute_metrics, compute_avg_distance
from utils.Visualise import create_tap_vid




def validate(model, loader, writer: SummaryWriter, epoch, device):
    vis_all_n_epoch = 5
    model.eval()

    with torch.no_grad():
        n = 0
        thr_acc, occ, jac = 0, 0, 0

        for i, (frames, trajs, vsbls, qrs) in enumerate(loader):

            trajs_pred, vis_pred, confidence_pred = model(frames.to(device), qrs.to(device))
            trajs_pred, vis_pred, confidence_pred = trajs_pred.cpu(), vis_pred.cpu(), confidence_pred.cpu()


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

            if (epoch % vis_all_n_epoch) == 4:
                if (i % 10) == 0:
                    video = create_tap_vid(pred)
                    writer.add_video(f"TapVidDavis_{i}", video[None, :, :, :, :], epoch, fps=10)

        writer.add_scalar("AVGThreshold", thr_acc/n, epoch)
        writer.add_scalar("OCCAccuracy", occ/n, epoch)
        writer.add_scalar("AVGJaccard", jac/n, epoch)

        return jac / n



def train(model, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device):
    steps_per_epoch = 500
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
            frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
            trajs_pred, vis_pred, confidence_pred = model(frames, qrs)
            loss = loss_fnct(trajs[0], vsbls[0], trajs_pred[0], vis_pred[0], confidence_pred[0], qrs[0])

            loss.backward()
            loss_sum += loss.item()
            iter += 1
            optimiser.step()

            #Validation
            if (steps % steps_per_epoch) == (steps_per_epoch-1):
                print(f"Running validation at step {steps} with loss {loss.item()}")
                epoch = steps // steps_per_epoch
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
    
#TODO: Add saving for optimiser and sceduler state
#TODO: Add usage of bfloat and gradient scaling

def main():
    lr = 0.00001
    wedecay = 0.00001
    batch_size = 1
    n_steps = 10_001

    seed = 42
    seed_everything(seed)

    now = datetime.now()
    now_str = now.strftime(("%d.%m.%Y_%H:%M:%S"))
    writer_dir = os.path.join("/scratch_net/biwidl304/amarugg/gluTracker/runs", now_str).replace("\\", "/" )
    os.mkdir(writer_dir)
    writer = SummaryWriter(log_dir=writer_dir)

    train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
    val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, prefetch_factor=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, prefetch_factor=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthTracker().to(device)
    model.cotracker.load_state_dict(torch.load("/scratch_net/biwidl304/amarugg/gluTracker/weights/depth_Tracker_final.pth", map_location=device))

    print(f"Loaded model of {sum(p.numel() for p in model.parameters())} Parameters")

    loss_fnct = track_loss_with_confidence

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wedecay, eps=1e-8)


    validate(model, val_loader, writer, -1, device)
    train(model, train_loader, val_loader, loss_fnct, optimiser, writer, n_steps, device)

    writer.close()


if __name__ == "__main__":
    main()
