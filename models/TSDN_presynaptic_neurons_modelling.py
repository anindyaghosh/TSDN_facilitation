import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import matlab_style_functions as msf
from utils import initialisations, IIR_Filter, naming_convention

"""Time constants and kernels"""
class params():

    def __init__(self):
        self.photo_z = {"b" : np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524]), 
                        "a" : np.array([1, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])}

        self.Ts = 0.001 # 1 ms timestep
        self.LPF5_TAU = 25 * self.Ts

        self.LPF_5 = {"b" : np.array([1 / (1 + 2*self.LPF5_TAU/self.Ts), 1 / (1 + 2*self.LPF5_TAU/self.Ts)]), 
                      "a" : np.array([1, (1 - 2*self.LPF5_TAU/self.Ts) / (1 + 2*self.LPF5_TAU/self.Ts)])}

        self.LPF5_K = np.exp(-1*0.05 / self.LPF5_TAU)

        self.LPFHR_TAU = 40 * self.Ts

        self.LPF_HR = {"b" : np.array([1 / (1 + 2*self.LPFHR_TAU/self.Ts), 1 / (1 + 2*self.LPFHR_TAU/self.Ts)]), 
                       "a" : np.array([1, (1 - 2*self.LPFHR_TAU/self.Ts) / (1 + 2*self.LPFHR_TAU/self.Ts)])}

        # Centre-surround kernel
        self.CSA_KERNEL = np.asarray([[-1, -1, -1],
                                      [-1, 8, -1], 
                                      [-1, -1, -1]]) * 1/9
        
        self.FDSR_TAU_FAST_ON = 0.5/50
        self.FDSR_TAU_FAST_OFF = 0.5/50
        
        self.FDSR_TAU_SLOW = 5.0/50
        
        self.FDSR_K_FAST_ON = np.exp(-1*self.Ts / self.FDSR_TAU_FAST_ON)
        self.FDSR_K_FAST_OFF = np.exp(-1*self.Ts / self.FDSR_TAU_FAST_OFF)
        
        self.FDSR_K_SLOW = np.exp(-1*self.Ts / self.FDSR_TAU_SLOW)
        
        # Spatial lateral inhibition kernel
        self.INHIB_KERNEL = 1.2 * np.array([[-1, -1, -1, -1, -1],
                                            [-1, 0, 0, 0, -1],
                                            [-1, 0, 2, 0, -1],
                                            [-1, 0, 0, 0, -1],
                                            [-1, -1, -1, -1, -1]])

class model_initialisation(params):
    def __init__(self, image_array_files, desired_resolution):
        self.image_array_files = image_array_files
        self.len_image_array = len(image_array_files)
        self.desired_resolution = desired_resolution
        super(model_initialisation, self).__init__()

    def initialisations(self):
        """
        Initialisations
        """
        
        # (480, 640)
        self.height, self.width = cv2.imread(self.image_array_files[0]).shape[:-1]
            
        self.degrees_in_image = self.desired_resolution[1]
        self.image_size, self.ds_size, self.H, self.pad_width = initialisations(self.degrees_in_image, (self.height, self.width))
        
        # Two channels to act as buffer i.e. -1 indexed channel is previous timestep's output
        self.on_f = np.zeros((2, *self.ds_size), dtype=np.float16)
        self.off_f = np.zeros_like(self.on_f)
        self.fdsr_on = np.zeros_like(self.on_f)
        self.fdsr_off = np.zeros_like(self.on_f)

        self.dbuffer1 = np.zeros((len(self.photo_z["b"]), *tuple(self.ds_size)))
        self.ONbuffer = np.zeros_like(self.dbuffer1)
        self.OFFbuffer = np.zeros_like(self.dbuffer1)
        
        self.EHR_buffer_right = np.zeros_like(self.dbuffer1)
        
        # vf = rf(pixels_to_keep, args.background, args.experiment_number, bg_contrast).run()

"""Early visual processing"""
class early_visual_processing(model_initialisation):
    def __init__(self):
        super(early_visual_processing, self).__init__()

    def model(self):
        
        # Extract green channel from BGR
        green = image[...,1]
        
        pixels_per_degree = self.image_size[1] / self.degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
        pixel2PR = int(pixels_per_degree) # ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling... )
        
        #     # Downsampled receptive fields
        #     Downsampledvf = cv2.resize(vf, np.flip(ds_size), interpolation=cv2.INTER_NEAREST)
        
        # Spatial filtered through LPF1
        sf = cv2.filter2D(green, -1, self.H)
        
        # Downsampled green channel
        DownsampledGreen = sf[::pixel2PR, ::pixel2PR]
        
        # Photoreceptor output after temporal band-pass filtering
        PhotoreceptorOut, self.dbuffer1 = IIR_Filter(self.photo_z["b"], self.photo_z["a"], DownsampledGreen/255, self.dbuffer1)
        
        # LMC output after spatial high pass filtering
        LMC_Out = msf.matlab_style_conv2(PhotoreceptorOut, self.CSA_KERNEL, mode='same', pad_width=self.pad_width)
        
        # Half-wave rectification
        # Clamp the high pass filtered data to separate the on and off channels
        self.on_f[1,...] = np.maximum(LMC_Out, 0.0)
        self.off_f[1,...] = -np.minimum(LMC_Out, 0.0)
        
        return self.on_f, self.off_f

"""Target-matched filtering"""    
class target_matched_filtering(model_initialisation):
    def __init__(self):
        super(target_matched_filtering, self).__init__()
        
    def fast_adaptive(self):
        
        # FDSR Implementation
        k_on = np.where((on_f[1,...] - on_f[0,...]) > 0.01, self.FDSR_K_FAST_ON, self.FDSR_K_SLOW)
        k_off = np.where((off_f[1,...] - off_f[0,...]) > 0.01, self.FDSR_K_FAST_OFF, self.FDSR_K_SLOW)

        # Apply low-pass filters to on and off channels
        self.fdsr_on[1,...] = ((1.0 - k_on) * on_f[1,...]) + (k_on * self.fdsr_on[0,...])
        self.fdsr_off[1,...] = ((1.0 - k_off) * off_f[1,...]) + (k_off * self.fdsr_off[0,...])
        
        return self.fdsr_on, self.fdsr_off
    
    def spatial_lateral_antagonism(self):
        
        # Subtract FDSR from half-wave rectified
        a_on = (on_f[1,...] - fdsr_on[1,...]).clip(min=0)
        a_off = (off_f[1,...] - fdsr_off[1,...]).clip(min=0)
        
        # Inhibition is implemented as spatial filter
        # Not true if this is a chemical synapse as this would require delays
        # Half-wave rectification added
        on_filtered = msf.matlab_style_conv2(a_on, self.INHIB_KERNEL, mode='same', pad_width=self.pad_width).clip(min=0)
        off_filtered = msf.matlab_style_conv2(a_off, self.INHIB_KERNEL, mode='same', pad_width=self.pad_width).clip(min=0)
        
        return on_filtered, off_filtered
        
    def delay_and_correlate(self):
        
        # Delayed channels using z-transform
        On_Delayed_Output, self.ONbuffer = IIR_Filter(self.LPF_5["b"], self.LPF_5["a"], on_filtered, self.ONbuffer)
        Off_Delayed_Output, self.OFFbuffer = IIR_Filter(self.LPF_5["b"], self.LPF_5["a"], off_filtered, self.OFFbuffer)
        
        # Correlation between channels
        Correlate_ON_OFF = on_filtered * Off_Delayed_Output # delayed_off[...,t]
        Correlate_OFF_ON = off_filtered * On_Delayed_Output # delayed_on[...,t]
        
        RTC_Output = Correlate_ON_OFF + Correlate_OFF_ON
        ESTMD_Output = RTC_Output.clip(min=0)
        
        return ESTMD_Output
    
class directional_selectivity(model_initialisation):
    def __init__(self):
        super(directional_selectivity, self).__init__()
        
    def model(self, direction):
        
        # Delayed channels using z-transform for HR
        EHR_Delayed_Output_right, self.EHR_buffer_right = IIR_Filter(self.LPF_HR["b"], self.LPF_HR["a"], ESTMD_Output.copy(), self.EHR_buffer_right)
        
        if direction == 'right':
            # Correlate delayed channels
            EHR_right = (EHR_Delayed_Output_right[:,:-1] * ESTMD_Output[:,1:]) - (ESTMD_Output[:,:-1] * EHR_Delayed_Output_right[:,1:])
            
            return EHR_right
        
        elif direction == 'left':
            # Correlate delayed channels
            EHR_left = -(EHR_Delayed_Output_right[:,:-1] * ESTMD_Output[:,1:]) + (ESTMD_Output[:,:-1] * EHR_Delayed_Output_right[:,1:])
            
            return EHR_left
        
image_array_files = glob('../../STMD/4496768/STNS3/28/*.jpg')   
direction = 'left'     
model_inits = model_initialisation(image_array_files, desired_resolution=(72, 72))
model_inits.initialisations()

os.makedirs('STNS3_28/images', exist_ok=True)
os.makedirs('STNS3_28/ESTMD_Output', exist_ok=True)

with tqdm(total=len(image_array_files)) as pbar:
    for t, file in enumerate(image_array_files):
        
        pbar.update(1)
        
        image = cv2.imread(file)
        
        """ESTMD model early visual processing -- Wiederman et al. (2008)"""
        on_f, off_f = early_visual_processing.model(model_inits)
        
        if t > 0:
            
            """ESTMD model target matching -- Wiederman et al. (2008)"""
            
            fdsr_on, fdsr_off = target_matched_filtering.fast_adaptive(model_inits)
            on_filtered, off_filtered = target_matched_filtering.spatial_lateral_antagonism(model_inits)
            
            ESTMD_Output = target_matched_filtering.delay_and_correlate(model_inits)
            
            """Directionally-selective ESTMD"""
            
            dESTMD_Output = directional_selectivity.model(model_inits, direction)
        
            """LPTC model"""
            
            cv2.imwrite(f'STNS3_28/images/{naming_convention(t+1)}.png', image)
            plt.imsave(f'STNS3_28/ESTMD_Output/{naming_convention(t+1)}.png', ESTMD_Output, dpi=500)
            
            # Buffer updates
            on_f[0,...] = on_f[1,...]
            off_f[0,...] = off_f[1,...]
            
            fdsr_on[0,...] = fdsr_on[1,...]
            fdsr_off[0,...] = fdsr_off[1,...]