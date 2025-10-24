import cv2
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import matlab_style_functions as msf

def naming_convention(i):
    # zfill pads string with zeros from leading edge until len(string) = 6
    return 'IMG' + str(i).zfill(6)

# Calcalute values required to initialise buffers and Gaussian kernel
def initialisations(degrees_in_image, image_size):
    pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
    pixel2PR = int(pixels_per_degree)
    ds_size = (np.asarray(image_size) / pixel2PR).astype(int)
    
    sigma = 1.4 / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixel = sigma * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree
    kernel_size = int(6 * sigma_pixel - 1)
    pad_width = int(kernel_size / 2)
    
    # Gaussian kernel
    H = msf.matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma_pixel)
    
    return image_size, ds_size, H, pad_width, pixel2PR

# IIR temporal band-pass filter
def IIR_Filter(b, a, Signal, dbuffer):
    dbuffer[:-1,...] = dbuffer[1:,...]
    dbuffer[-1,...] = np.zeros(dbuffer[-1,...].shape)
    
    for k in range(len(b)):
        dbuffer[k,...] += (Signal * b[k])
        if k <= (len(b)-2):
            dbuffer[k+1,...] -= (dbuffer[0,...] * a[k+1])
    
    Filtered_Data = dbuffer[0,...]
    
    return Filtered_Data, dbuffer

# Plotting function
def output_images(image, image_buffer, t, delay, pixel2PR, bbox, output_folder):
    image[image < 0.0] = 0
    image_output_norm = cv2.normalize(image, -1)
    idx = np.where(image_output_norm > 0)
    pixel_intensities = image_output_norm[idx]
    
    if bbox is not None:
        x, y, width, height = tuple(map(lambda x: round(x/pixel2PR), bbox))
        
        rect = plt.Rectangle((x+1.5, y-0.5), width-1, height-1, fill=False, color="limegreen", linewidth=1)
    
        fig, ax = plt.subplots()
        ax.imshow(image_output_norm, cmap='gray')
        ax.add_patch(rect)
    
    fig, ax = plt.subplots()
    ax.imshow(image_buffer[...,[2,1,0],-1].astype(np.uint8))
    _patches = [plt.Circle(((c[1]-0.5)*pixel2PR, (c[0]+1.5)*pixel2PR), radius=5, color='r', alpha=p) for p,c in zip(pixel_intensities, np.column_stack(idx))]
    collection = PatchCollection(_patches, match_original=True)
    ax.add_collection(collection)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(f'{output_folder}/{naming_convention(t-delay+1)}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # return _patches