import torch
from dataset.Dataloader import TapvidDavisFirst, TapData
from models.Model import GluTracker
from utils.Visualise import display_tap_vid

def main():
    batch_size = 1
    val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GluTracker().to(device)
    model.load()
    model.eval()

    with torch.no_grad():

        for i, (frames, trajs, vsbls, qrs) in enumerate(val_loader):
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

            trajs_pred[:,:,:,0] *= frames.shape[4]
            trajs_pred[:,:,:,1] *= frames.shape[3]

            pred = TapData(
                frames,
                trajs_pred,
                vis_pred,
                qrs
            )
            display_tap_vid(pred)


if __name__ == "__main__":
    main()
