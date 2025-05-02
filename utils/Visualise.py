import matplotlib.pyplot as plt
import numpy as np
import random
from dataset.Dataloader import TapData

def display_tap_vid(sample: TapData, save=False, idx=0):
        video = sample.video.numpy()[idx]
        trajectory = sample.trajectory.numpy()[idx]
        visibility = sample.visibility.numpy()[idx]

        output_dir = "/scratch_net/biwidl304/amarugg/gluTracker/media"
        seed = 42
        S, C, H, W = video.shape
        N = trajectory.shape[1]

        random.seed(seed)
        cmap = {
            n: (random.random(), random.random(), random.random())
            for n in range(N)
        }

        for s in range(S):
            frame = np.transpose(video[s],(1,2,0)) # C, H, W -> H, W, C
            frame = frame.astype(int)

            plt.figure(figsize=(H / 100, W / 100), dpi=100)
            plt.imshow(frame)
            plt.axis('off')

            for n in range(N):
                x, y = trajectory[s, n]
                is_visible = visibility[s, n].item() > 0.5
                color = cmap[n]

                plt.scatter(
                    x, y,
                    c=[color],
                    alpha=1.0 if is_visible else 0,
                    s=20,
                    edgecolors='black'
                )
            if save:
                plt.savefig(f"{output_dir}/frame{s}.png", bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()