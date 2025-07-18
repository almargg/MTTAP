import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
from models.Model import DepthTracker, DepthTrackerOnline
from models.utils.Loss import  track_loss_with_confidence
from utils.Metrics import compute_metrics, compute_avg_distance
from utils.Visualise import create_tap_vid, vis_against_gt
from models.utils.Loss import  track_loss_with_confidence





def validate(model, loader, device):
    model.eval()

    with torch.no_grad():
        n = 0
        thr_acc, occ, jac = 0, 0, 0
        loss_sum  = 0

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

            loss = track_loss_with_confidence(trajs[0], vsbls[0], trajs_pred[0], vis_pred[0], confidence_pred[0], qrs[0])
            print(f"Loss: {loss}")

            vis_against_gt(pred, gt)
            avg_thrh_acc, avg_occ_acc, avg_jac = compute_metrics(gt, pred)
            thr_acc += avg_thrh_acc
            occ += avg_occ_acc
            jac += avg_jac
            n += 1

        print("AVGThreshold", thr_acc/n )
        print("OCCAccuracy", occ/n)
        print("AVGJaccard", jac/n)

val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, prefetch_factor=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DepthTracker().to(device)
model.cotracker.load_state_dict(torch.load("/scratch_net/biwidl304/amarugg/gluTracker/weights/depth_Tracker_final.pth", map_location=device))
#model.load()
#model.to(device)

print(f"Loaded model of {sum(p.numel() for p in model.parameters())} Parameters")

validate(model, val_loader, device)
