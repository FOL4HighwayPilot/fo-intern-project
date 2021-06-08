import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *


# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

# For every mask image
for mask_name in tqdm.tqdm(mask_list):

    # Access required folders
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name)
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)

    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
    cpy_image  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

    # Write output image into IMAGE_OUT_DIR folder
    cv2.imwrite(image_out_path, opac_image)

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:
        plt.imshow(opac_image)
        plt.show()
