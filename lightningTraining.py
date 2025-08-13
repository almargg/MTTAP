import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from models.Model import OnlineTracker, CoTrackerThreeOnline
from models.utils.Loss import track_loss_with_confidence
from utils.Metrics import compute_metrics
from dataset.Dataloader import KubricDataset, TapvidDavisFirst, TapData

NUM_NODES = 1  # Set to the number of nodes you want to use
MIXED_PRECISION = True  # Set to True if you want to use mixed precision training
RESUME_TRAINING = False  # Set to True if you want to resume training from a checkpoint
IS_SANITY_CHECK = False  # Set to True if you want to run a sanity check

class CoTrackerDataset(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = KubricDataset("/srv/beegfs02/scratch/alex_project/data/kubric/kubric_movi_f_120_frames_dense/movi_f")
        #self.train_dataset = KubricDataset("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/kubric/kubric_movi_f_120_frames_dense/movi_f")
        self.val_dataset = TapvidDavisFirst("/scratch_net/biwidl304_second/amarugg/kubric_movi_f/tapvid/tapvid_davis/tapvid_davis.pkl")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, prefetch_factor=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, prefetch_factor=2)
    
    #def state_dict(self):
    #    return super().state_dict()
    
    #def load_state_dict(self, state_dict):
    #    super().load_state_dict(state_dict)
    


class LightningTraining(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.save_hyperparameters()
        #TODO: Try to compile the model

        #self.model = CoTrackerThreeOnline(stride=4, corr_radius=3, window_len=16)
        #load fnet and freeze it
        #self.model.fnet.load_state_dict(torch.load("/scratch_net/biwidl304/amarugg/gluTracker/weights/fnet.pth"))
        #self.model.fnet.eval()
        self.model = OnlineTracker()
        self.model.load()
        #freeze layers of fnet
        for param in self.model.fnet.parameters():
            param.requires_grad = False

        self.loss_fn = track_loss_with_confidence
        self.lr = 2e-5#0.0001 #5e-8
        self.wd = 0.00001#1e-6

    def forward(self, frames, qrs): 
        return self.model.track_vid(frames, qrs)
        #trajs_pred, vis_pred, conf_pred, train = self.model(frames, qrs)
        #return trajs_pred, vis_pred, conf_pred

    def training_step(self, batch, batch_idx):
        frames, trajs, vsbls, qrs = batch

        # Skip batches with less than 128 tracks
        #if trajs.shape[2] < 128:
        #    print(f"Skipping batch with {trajs.shape[2]} tracks, less than 128", flush=True)
        #    return None
        trajs_pred, vis_pred, confidence_pred = self(frames, qrs)
        
        # Assuming trajs and vsbls are the ground truth tensors
        loss = self.loss_fn(trajs[0], vsbls[0], trajs_pred[0], vis_pred[0], confidence_pred[0], qrs[0])

        self.log('train_loss', loss)

        return loss
        

    def validation_step(self, batch, batch_idx):
        #if self.trainer.global_rank != 0:
        #    print(f"Skipping validation step for non-master process with idx {batch_idx}", flush=True)
        #    return
        frames, trajs, vsbls, qrs = batch
        trajs_pred, vis_pred, confidence_pred = self(frames, qrs)
        
        # Assuming trajs and vsbls are the ground truth tensors
        loss = self.loss_fn(trajs[0], vsbls[0], trajs_pred[0], vis_pred[0], confidence_pred[0], qrs[0])

        self.log('val_loss', loss, sync_dist=True)

        gt = TapData(
            frames.cpu(),
            trajs.cpu(),
            vsbls.cpu(),
            qrs.cpu()
        )
        pred = TapData(
            frames.cpu(),
            trajs_pred.cpu(),
            vis_pred.cpu(),
            qrs.cpu()
        )

        avg_thrh_acc, avg_occ_acc, avg_jac = compute_metrics(gt, pred)
        values = {
            'avg_thrh_acc': avg_thrh_acc,
            'avg_occ_acc': avg_occ_acc,
            'avg_jac': avg_jac
        }
        self.log_dict(values, sync_dist=True)
        
        return loss
    
    
    def configure_optimizers(self):

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd, eps=1e-8)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
        n_steps = 200000 
        print(f"Estimated stepping batches: {n_steps}", flush=True)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=n_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
        
    
    def lr_scheduler_step(self, scheduler, optimiser_idx=None, metric=None):
        # This method is called at the end of each training step
        scheduler.step()


    #def on_after_backward(self):
        # Log gradients after backward pass
        #for name, param in self.model.named_parameters():
        #    if param.grad is not None:
        #        self.logger.experiment.add_histogram(f'gradients/{name}', param.grad, self.global_step)


        
        
# Set to mixed
#Check gpu generation to see if we can use bfloat16
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    float_precision = 'bf16'
else:
    float_precision = 16

if IS_SANITY_CHECK:
    trainer = pl.Trainer(overfit_batches=1)

trainer = pl.Trainer(
    strategy=DDPStrategy(find_unused_parameters=False),
    devices=-1 if torch.cuda.is_available() else 1,
    num_nodes=NUM_NODES,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision= float_precision if (MIXED_PRECISION and torch.cuda.is_available()) else 32,
    default_root_dir="/scratch_net/biwidl304/amarugg/gluTracker/weights",
    val_check_interval= 1000, #After how many training steps to run validation
    max_epochs=50,
    callbacks=[
        #EarlyStopping(
        #    monitor='avg_jac',
        #    mode='max',
        #    patience=12, # Roughly 2 epochs
        #    verbose=True
        #),
        LearningRateMonitor(logging_interval='step'),
        #ModelCheckpoint(
        #    monitor="val_loss",  # Track validation loss
        #    mode="min",        # Smaller validation loss is better
        #    save_top_k=2,      # Save the top 2 models
        #    dirpath="checkpoints/", # Directory to save checkpoints
        #    filename="model-{epoch:02d}-{val_loss:.2f}"  # Filename format
        #),
    ],
    
)


#TODO: Try Accumulated Gradients, Gradient Clipping, Stochastic Weight Average, learning rate finder
#TODO: Add argument parser instead of defines
#TODO: ADD auto wall-time resubmission
#TODO: ADD automatic checkpointing ony every epoch
model = LightningTraining()
dm = CoTrackerDataset(batch_size=1)


# Run initial validation if we are on GPU and do real training
if torch.cuda.is_available():
    trainer.validate(model, datamodule=dm)

if RESUME_TRAINING:
    trainer.fit(model, datamodule=dm, ckpt_path="/scratch_net/biwidl304/amarugg/gluTracker/weights/checkpoint.ckpt")
else:
    trainer.fit(model, datamodule=dm)
