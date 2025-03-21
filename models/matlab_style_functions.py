import numpy as np
from scipy import signal

def matlab_style_gauss2D(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', [shape], [sigma])
    """
    m, n = [(ss-1)/2 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def matlab_style_conv2(x, y, mode='same', **kwargs):
    """
    should give the same result as MATLAB's
    conv2(x, y, mode='same') with padding
    """
    pad_width = kwargs.pop('pad_width', 0)
    # Add padding
    padded = np.pad(x, pad_width=pad_width, mode='reflect')
    convolved = np.rot90(signal.convolve2d(np.rot90(padded, 2), np.rot90(y, 2), mode=mode), 2)
    
    # Remove padding
    return convolved[pad_width:-pad_width, pad_width:-pad_width]