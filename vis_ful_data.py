import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    :param quaternion: A numpy array of shape (4,) representing the quaternion (x, y, z, w).
    :return: A numpy array of shape (3, 3) representing the rotation matrix.
    """
    r = Rotation.from_quat(quaternion)
    return r.as_matrix()

data_root = "/home/amarugg/Downloads/0000"

npy_path = os.path.join(data_root, "0000.npy")
frames_path = os.path.join(data_root, "frames")
depth_dir = os.path.join(data_root, "depths")


depth_imgs = []
for f in os.listdir(depth_dir):
    depth_imgs.append(np.load(os.path.join(depth_dir, f), allow_pickle=True))

m_depth = np.stack(depth_imgs)
print(f"Depth images shape: {m_depth.shape}")
print(f"Depth images dtype: {m_depth.dtype}")
print(f"Depth images min: {np.min(m_depth)}, max: {np.max(m_depth)}")


annot_dict = np.load(npy_path, allow_pickle=True).item()

trajs = annot_dict["coords"]
visibility = annot_dict["visibility"]
traj_depth = annot_dict["coords_depth"]
depth = annot_dict["depth"]
segmentation = annot_dict["segmentations"]
camera = annot_dict["camera"]

# Create 3D trajectory from camera positions
camera_positions = camera["positions"]
camera_quaternions = camera["quaternions"]
camera_rotations = np.array([quaternion_to_rotation_matrix(q) for q in camera_quaternions])

# Coordinates of the camera model in the camera frame in homogeneous coordinates
camera_model = np.array([[-0.1, -0.1, -0.1, 1], [-0.1, 0.1, -0.1, 1], [0.1, 0.1, -0.1, 1], [0.1, -0.1, -0.1, 1], [0, 0, 0, 1]])

start_points = np.stack([camera_model[0, :], camera_model[1, :], camera_model[2, :], camera_model[3, :], camera_model[0, :], camera_model[1, :], camera_model[2, ], camera_model[3, :]])
end_points = np.stack([camera_model[1, :], camera_model[2, :], camera_model[3, :], camera_model[0, :], camera_model[4, :], camera_model[4, :], camera_model[4, :], camera_model[4, :]])




#TODO
R0 = camera_rotations[0]
start_pt = (R0 @ camera_positions[0].T).T

# Convert into coordinate frame of first camera
#camera_positions = (R0 @ camera_positions.T).T - start_pt
#camera_rotations = R0 @ camera_rotations

#x = camera_positions[:, 0]
#y = camera_positions[:, 1]
#z = camera_positions[:, 2]
#idxs = np.linspace(0, len(x) - 1, len(x), dtype=int)


img_paths = os.listdir(frames_path)
rgbs = np.stack([
            cv2.cvtColor(cv2.imread(os.path.join(frames_path, im)), cv2.COLOR_BGR2RGB)
            for im in img_paths
        ])

depth_min = np.min(depth)
depth_max = np.max(depth)
n_objects = np.max(segmentation) + 1

print(f"Depth min: {depth_min}, Depth max: {depth_max}")

# Paint each indx of the segmentation with a different color
segmentation_colored = np.zeros(segmentation.shape[:-1] + (3,), dtype=np.uint8)
for i in range(n_objects):
    mask = (segmentation == i)
    #Repeat the mask for each channel
    mask = np.repeat(mask, 3, axis=3)
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    l = segmentation_colored[mask].shape[0] // 3
    list = color.tolist()* l
    l_col = np.array(list, dtype=np.uint8)
    segmentation_colored[mask] = l_col

#Visualize depth maps
norm_depth = depth / np.max(depth)  # Normalize depth to [0, 1]
#Cast depth to range blue to red
depth_maps = np.stack([
    cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    for d in norm_depth
])

#Select a subset of the trajectory for visualization
# Randomly select 512 points from the trajectory
np.random.seed(42)
indices = np.random.choice(trajs.shape[0], size=512, replace=False)
trjs = trajs[indices, :]
vsbls = ~visibility[indices, :]



def generate_colors(n):
    colors = []
    for i in range(n):
        hue = (240*i) / (n-1)
        hsv_color = np.uint8([[[hue * 179 // 360, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))
    return colors

colors = generate_colors(trjs.shape[0])

center_points = []
for i in range(depth_maps.shape[0]):
    """
    fig = plt.figure()
    traj = trajs.astype(np.int16)
    min = np.min(traj)
    max = np.max(traj)

    mask_x = np.logical_and(traj[:,i,0] >= 0, traj[:,i,0] < 512)
    mask_y = np.logical_and(traj[:,i,1] >= 0 , traj[:,i,1] < 512)
    mask = np.logical_and(mask_x,  mask_y)

    depths = depth[i][traj[:, i, 0][mask], traj[:, i, 1][mask]]
    vsbls = visibility[:,i][mask]

    depths = depths[~vsbls]
    metric_depth = traj_depth[:,i][mask][~vsbls]

    #randomly sample 1000 points
    

    #normalised_depth = depths / np.max(depths)  # Normalize depth to [0, 1]
    #normalised_metric_depth = metric_depth / np.max(metric_depth)  # Normalize metric depth

    plt.scatter(depths, metric_depth, s=1, c='b', label='Depth')
    plt.xlabel('Normalised Depth')
    plt.ylabel('Normalised Metric Depth')
    plt.title(f'Depth vs Metric Depth for Frame {i}')
    plt.legend()
    
    fig.canvas.draw()
    corr_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    corr_img = corr_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    corr_img = cv2.cvtColor(corr_img, cv2.COLOR_RGB2BGR)
    corr_img = cv2.resize(corr_img, (512, 512))  # Resize for better visualization    

    plt.close(fig)
    """
    #stack images
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origin = camera_positions[i]
    R = camera_rotations[i]
    #Create essential matrix by stacking R | origin
    E = np.hstack((R, origin.reshape(3, 1)))
    
    sp = (E @ start_points.T).T 
    ep = (E @ end_points.T).T 
    #dist = np.linalg.norm(ep - sp, axis=1)
    for j in range(sp.shape[0]):
        if j == 6:
            c = 'g'
        else:
            c = 'b'
        ax.plot([sp[j,0], ep[j,0]], [sp[j,1], ep[j,1]], [sp[j,2], ep[j,2]], color=c, linewidth=0.5)

    #msk = idxs <= i
    center_points.append(ep[4])

    trajectory = np.stack([center_points])[0]
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    ax.scatter(x, y, z, c='r', marker='o', s=1)

    # Set Labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Camera Trajectory')

    #Set limits
    center = ep[4]
    ax.set_xlim([-1+center[0], 1+center[0]])
    ax.set_ylim([-1+center[1], 1+center[1]])
    ax.set_zlim([-1+center[1], 1+center[1]])    

    R_t = R.T
    beta = -np.arcsin(R_t[2, 0])  # Rotation around x-axis 
    alpha = np.arctan2(R_t[2, 1]/np.cos(beta), R_t[2, 2]/np.cos(beta))  # Rotation around y-axis
    gamma = np.arctan2(R_t[1, 0]/np.cos(beta), R_t[0, 0]/np.cos(beta))  # Rotation around z-axis

    beta = beta * 180 / np.pi
    alpha = alpha * 180 / np.pi
    gamma = gamma * 180 / np.pi
    
    # Rotate the view   
    ax.view_init(elev=-90, azim=0, roll=180)
    
    # Convert figure to cv2 image
    fig.canvas.draw()
    fig_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)
    fig_img = cv2.resize(fig_img, (512, 512))  # Resize for better visualization    
    plt.close(fig)
    
    def paint_frame(frame, points, visibility, colors):
        circle_radius = 5
        circle_thickness = 2

        for i in range(points.shape[0]):
            x, y = int(points[i,0]), int(points[i,1])
            if visibility[i] > 0.5:
                thickness = -1
            else:
                thickness = circle_thickness
            cv2.circle(frame, (x,y), circle_radius, colors[i], thickness)
        return frame
    
    # Overlay tracks on the RGB image
    img = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
    img = paint_frame(img.copy(), trjs[:,i,:], vsbls[:,i], colors)
    

    depth_img = depth_maps[i]
    
    segmentation_img = segmentation_colored[i]          

    stacked = np.hstack((img, depth_img, segmentation_img, fig_img))

    cv2.imshow('RGB, Depth, Segmentation and track', stacked)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    cv2.imwrite(os.path.join(data_root, f"frame_{i:04d}.png"), stacked)
    
cv2.destroyAllWindows()
