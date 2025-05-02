import torch
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData
from models.Model import GluTracker
from models.utils.Loss import tap_loss, coTrack_loss
from utils.Metrics import compute_metrics, compute_avg_distance



def train(model, loader, loss_function, optimiser, device):
    model.train()
    total = 0
    running_loss = 0
    avg_dist = 0
    for i, (frames, trajs, vsbls, qrs) in enumerate(loader):
        frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
        optimiser.zero_grad()
        loss = 0
        for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
            for j in range(frames.shape[1] - 1):
                trajs_ = trajs[b, j, :, :] # N, 2
                qrs_msk = qrs[b,:,0] <= j #N
                queries = trajs_[qrs_msk]

                frames_ = torch.stack([frames[b,j,:,:,:], frames[b,j+1,:,:,:]], dim=0)

                queries = torch.unsqueeze(queries, 0)
                frames_ = torch.unsqueeze(frames_, 0)

                #Make sure we have queries to track
                if queries.shape[1] == 0:
                    continue

                pred_pose, pred_vis = model(queries, frames_)

                gt_traj = trajs[b, j+1, :, :][qrs_msk] 
                avg_dist += compute_avg_distance(gt_traj, pred_pose)


                loss += loss_function(torch.unsqueeze(trajs[b, j+1, :, :][qrs_msk, 0:], 0), torch.unsqueeze(vsbls[b, j+1, :][qrs_msk],0), pred_pose, pred_vis)
                
                running_loss += loss.item()
                total += 1
                if torch.isnan(loss):
                    print(f"Loss became nan at frame: {j} in batch {b}")

        loss.backward()
        optimiser.step()

        if i == 1000:
            print(f"Average pixel distance while training was {(avg_dist/(i+1))}")
            break

    return running_loss / total

def validate(model, loader, device):
    model.eval()

    with torch.no_grad():
        n = 0
        thr_acc, occ, jac = 0, 0, 0

        for i, (frames, trajs, vsbls, qrs) in enumerate(loader):
            frames, trajs, vsbls, qrs = frames.to(device), trajs.to(device), vsbls.to(device), qrs.to(device)
            trajs_pred = torch.zeros_like(trajs)
            vis_pred = torch.zeros_like(vsbls)

            for b in range(frames.shape[0]): #Run each batch seperately as dimensions of tracked points are not guaranteed to match
                for j in range(frames.shape[1] - 1):

                    new_ptns = qrs[b,:,0] == j #K
                    trajs_pred[b,j,:,:][new_ptns] = trajs[b,j,:,:][new_ptns] # Add new tracks

                    qrs_msk = qrs[b,:,0] <= j #K
                    queries = torch.unsqueeze(trajs_pred[b, j, :, :][qrs_msk], 0)

                    frames_ = torch.stack([frames[b,j,:,:,:], frames[b,j+1,:,:,:]], dim=0)
                    frames_ = torch.unsqueeze(frames_, 0)

                    pred_pose, pred_vis = model(queries, frames_)
                    
                    pred_pose, pred_vis = pred_pose.detach(), pred_vis.detach()
                    trajs_pred[b,j+1,:,:][qrs_msk] = pred_pose[0,:,:]
                    vis_pred[b,j+1,:][qrs_msk] = pred_vis[0,:]

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
        print(f"Average threshold accuracy: {thr_acc/n}")
        print(f"Occlusion accuracy: {occ/n}")
        print(f"Average Jaccard: {jac/n}")


def main():
    lr = 5e-5
    batch_size = 1
    epochs = 30

    train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
    val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GluTracker().to(device)

    print(f"Loaded model of {sum(p.numel() for p in model.parameters())} Parameters")

    loss_fnct = coTrack_loss

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    #model.load()
    #optimiser = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        avg_loss = train(model, train_loader, loss_fnct, optimiser, device)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        validate(model, val_loader, device)
    model.save()


if __name__ == "__main__":
    main()
