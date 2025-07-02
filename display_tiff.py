import cv2
import os
import numpy as np
def fix_tiff(tiff_path):
    """
    Fix a TIFF image by replacing values greater than 10000 with the average of surrounding pixels.
    
    Args:
        tiff_path (str): Path to the TIFF file.
    """
    # Read the TIFF image
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read the image from {tiff_path}")
        return
    
    n_max = np.max(img)
    n_min = np.min(img)

    changes = False

    while n_max > 10000:
        changes = True
        print(f"Max value {n_max} is too high, replacing with average of surrounding pixels.")
        max_loc = np.unravel_index(np.argmax(img, axis=None), img.shape)    
        # Replace max with average of surrounding pixels
        # Ensure we don't go out of bounds
        right = np.min(np.array([max_loc[0] + 2, img.shape[0]]))
        left = np.max(np.array([max_loc[0] - 1, 0]))
        bottom = np.min(np.array([max_loc[1] + 2, img.shape[1]]))
        top = np.max(np.array([max_loc[1] - 1, 0]))
        surrounding_pixels = img[left:right, top:bottom]
        surrounding_pixels = surrounding_pixels[surrounding_pixels != n_max]  # Exclude the max value itself
        if surrounding_pixels.size > 0:
            avg_surrounding = np.mean(surrounding_pixels)
            img[max_loc] = avg_surrounding
        n_max = np.max(img)
        n_min = np.min(img)
    if changes:
        cv2.imwrite(tiff_path, img)  # Save the fixed image back to disk

def fix_depth_images(directory_path):
    """
    Fix all TIFF images in a directory by replacing values greater than 10000 with the average of surrounding pixels.
    
    Args:
        directory_path (str): Path to the directory containing TIFF files.
    """
    for folder in sorted(os.listdir(directory_path)):
        print(f"Processing folder: {folder}")
        for i in range(120):
            file= f"depth_{i:05d}.tiff" #convert to string with leading zeros
            tiff_file_path = os.path.join(directory_path,folder, file)
            fix_tiff(tiff_file_path)
            


def display_tiff(tiff_path, rgb_path):
    """
    Display a TIFF image using OpenCV.
    
    Args:
        tiff_path (str): Path to the TIFF file.
    """
    # Read the TIFF image
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read the image from {tiff_path}")
        return
    
    rgba_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    if rgba_img is None:
        print(f"Error: Could not read the RGB image from {rgb_path}")
        return   
    bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGR)
    
    n_max = np.max(img)
    n_min = np.min(img)
    
    img_normalized = (img - n_min) / (n_max - n_min) * 255
    img_normalized = img_normalized.astype(np.uint8)
    colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
    
    img_stacked = np.hstack((colored, bgr_img))
    # Display the image
    cv2.imshow('TIFF Image', img_stacked)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Example usage
if __name__ == "__main__":
    #directory_path = "/scratch-second/amarugg/generated_data/Split0/00003"
    #directory_path = "/scratch-second/amarugg/generated_data/Split0/00000"  
    base_path = "/scratch-second/amarugg/generated_data/Split2"
    #for i in range(120):
    #        file= f"depth_{i:05d}.tiff" #convert to string with leading zeros
    #        png= f"rgba_{i:05d}.png"
    #        tiff_file_path = os.path.join(directory_path, file)
    #        rgb_file_path = os.path.join(directory_path, png)
    #        display_tiff(tiff_file_path, rgb_file_path)
    fix_depth_images(base_path)
