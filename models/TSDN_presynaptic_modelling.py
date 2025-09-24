import cv2
import matlab_style_functions as msf
import numpy as np
from scipy import signal
from tqdm import tqdm
import utils

image_array = []
desired_resolution = image_array[0].shape * 0.5

"""Time constants and kernels"""

photo_z = {"b" : np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524]), 
           "a" : np.array([1, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])}

Ts = 0.001 # 1 ms timestep
LPF5_TAU = 25 * Ts
LPF_5 = {"b" : np.array([1 / (1+2*LPF5_TAU/Ts), 1 / (1+2*LPF5_TAU/Ts)]), 
         "a" : np.array([1, (1-2*LPF5_TAU/Ts) / (1+2*LPF5_TAU/Ts)])}

LPF5_K = np.exp(-1*0.05 / LPF5_TAU)

LPFHR_TAU = 40 * Ts

LPF_HR = {"b" : np.array([1 / (1+2*LPFHR_TAU/Ts), 1 / (1+2*LPFHR_TAU/Ts)]), 
          "a" : np.array([1, (1-2*LPFHR_TAU/Ts) / (1+2*LPFHR_TAU/Ts)])}

# Centre-surround kernel
CSA_KERNEL = np.asarray([[-1, -1, -1],
                         [-1, 8, -1], 
                         [-1, -1, -1]]) * 1/9

FDSR_TAU_FAST_ON = 0.5/50 * Ts*1000
FDSR_TAU_FAST_OFF = 0.5/50 * Ts*1000

FDSR_TAU_SLOW = 5.0/50 * Ts*1000

FDSR_K_FAST_ON = np.exp(-1*Ts / FDSR_TAU_FAST_ON)
FDSR_K_FAST_OFF = np.exp(-1*Ts / FDSR_TAU_FAST_OFF)

FDSR_K_SLOW = np.exp(-1*Ts / FDSR_TAU_SLOW)

# Spatial lateral inhibition kernel
INHIB_KERNEL = 1.2 * np.array([[-1, -1, -1, -1, -1],
                                [-1, 0, 0, 0, -1],
                                [-1, 0, 2, 0, -1],
                                [-1, 0, 0, 0, -1],
                                [-1, -1, -1, -1, -1]])

"""
Initialisations
"""

def initialisations(degrees_in_image):
    image_size = image_array[0].shape[:-1]
    pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
    pixel2PR = int(pixels_per_degree)
    ds_size = (tuple(int(np.ceil(x/pixel2PR)) for x in image_size))
    
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
    
degrees_in_image = desired_resolution[1]
image_size, ds_size, H, pad_width = initialisations(degrees_in_image)

on_f = np.zeros((len(image_array), *ds_size), dtype=np.float16)
off_f = np.zeros_like(on_f)
fdsr_on = np.zeros_like(on_f)
fdsr_off = np.zeros_like(on_f)
ESTMD_Output = np.zeros_like(on_f)
RTC_Output = np.zeros_like(on_f)

vf = rf(pixels_to_keep, args.background, args.experiment_number, bg_contrast).run()

with tqdm(total=len(image_array)) as pbar:
    for t, image in enumerate(image_array):
        
        pbar.update(1)
        
        """ESTMD model Wiederman et al. (2008)"""
        
        # Extract green channel from BGR
        green = image[:,:,1]
        
        image_size = green.shape
        pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
        pixel2PR = int(pixels_per_degree) # ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling... )
        
        # Downsampled receptive fields
        Downsampledvf = cv2.resize(vf, np.flip(ds_size), interpolation=cv2.INTER_NEAREST)
            
        # Spatial filtered through LPF1
        sf = cv2.filter2D(green, -1, H)
        
        # Downsampled green channel
        DownsampledGreen = cv2.resize(sf, np.flip(ds_size), interpolation=cv2.INTER_NEAREST)
        
        try:
            dbuffer1
        except NameError:
            dbuffer1 = np.zeros((len(photo_z["b"]), *DownsampledGreen.shape))
        
        # Photoreceptor output after temporal band-pass filtering
        PhotoreceptorOut, dbuffer1 = IIR_Filter(photo_z["b"], photo_z["a"], DownsampledGreen/255, dbuffer1)
        
        # LMC output after spatial high pass filtering
        LMC_Out = msf.matlab_style_conv2(PhotoreceptorOut, CSA_KERNEL, mode='same', pad_width=pad_width)
        
        # Half-wave rectification
        # Clamp the high pass filtered data to separate the on and off channels
        on_f[t,:,:] = np.maximum(LMC_Out, 0.0)
        off_f[t,:,:] = -np.minimum(LMC_Out, 0.0)
        
        if t > 0:
            # FDSR Implementation
            k_on = np.where((on_f[t,:,:] - on_f[t-1,:,:]) > 0.01, FDSR_K_FAST_ON, FDSR_K_SLOW)
            k_off = np.where((off_f[t,:,:] - off_f[t-1,:,:]) > 0.01, FDSR_K_FAST_OFF, FDSR_K_SLOW)
            
            # Apply low-pass filters to on and off channels
            fdsr_on[t,:,:] = ((1.0 - k_on) * on_f[t,:,:]) + (k_on * fdsr_on[t-1,:,:])
            fdsr_off[t,:,:] = ((1.0 - k_off) * off_f[t,:,:]) + (k_off * fdsr_off[t-1,:,:])
            
            # Subtract FDSR from half-wave rectified
            a_on = (on_f[t,:,:] - fdsr_on[t,:,:]).clip(min=0)
            a_off = (off_f[t,:,:] - fdsr_off[t,:,:]).clip(min=0)
            
            # Inhibition is implemented as spatial filter
            # Half-wave rectification added
            on_filtered = msf.matlab_style_conv2(a_on, INHIB_KERNEL, mode='same', pad_width=pad_width).clip(min=0)
            off_filtered = msf.matlab_style_conv2(a_off, INHIB_KERNEL, mode='same', pad_width=pad_width).clip(min=0)
            
            try:
                ONbuffer
                OFFbuffer
            except NameError:
                ONbuffer = np.zeros((len(LPF_5["b"]), *on_filtered.shape))
                OFFbuffer = np.zeros((len(LPF_5["b"]), *off_filtered.shape))
            
            # Delayed channels using z-transform
            On_Delayed_Output, ONbuffer = IIR_Filter(LPF_5["b"], LPF_5["a"], on_filtered, ONbuffer)
            Off_Delayed_Output, OFFbuffer = IIR_Filter(LPF_5["b"], LPF_5["a"], off_filtered, OFFbuffer)
            
            # Correlation between channels
            Correlate_ON_OFF = on_filtered * Off_Delayed_Output
            Correlate_OFF_ON = off_filtered * On_Delayed_Output
            
            # ESTMD output
            RTC_Output[t,:,:] = (Correlate_ON_OFF + Correlate_OFF_ON)
            ESTMD_Output[t,:,:] = (RTC_Output[t,:,:]).clip(min=0)
            
    """LPTC model"""