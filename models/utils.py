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
    
    return image_size, ds_size, H, pad_width

# IIR temporal band-pass filter
def IIR_Filter(b, a, Signal, dbuffer):
    dbuffer[:-1,:,:] = dbuffer[1:,:,:]
    dbuffer[-1,:,:] = np.zeros(dbuffer[-1,:,:].shape)
    
    for k in range(len(b)):
        dbuffer[k,:,:] += (Signal * b[k])
        if k <= (len(b)-2):
            dbuffer[k+1,:,:] = dbuffer[k+1,:,:] - (dbuffer[0,:,:] * a[k+1])
    
    Filtered_Data = dbuffer[0,:,:]
    return Filtered_Data, dbuffer