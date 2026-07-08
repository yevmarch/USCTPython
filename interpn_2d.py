import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpn_2d(x0, y0, Z, x1, y1):
    #Equivalent of MATLAB's interpn for 2D gridded data (ndgrid convention)
    x0_np = cp.asnumpy(x0[:, 0])
    y0_np = cp.asnumpy(y0[0, :])
    Z_np = cp.asnumpy(Z)

    interpFunc = RegularGridInterpolator((x0_np, y0_np), Z_np, method='linear', bounds_error=False, fill_value=np.nan)

    x1_np = cp.asnumpy(x1)
    y1_np = cp.asnumpy(y1)
    pts = np.stack([x1_np.ravel(), y1_np.ravel()], axis=-1)
    result = interpFunc(pts).reshape(x1_np.shape)

    return cp.asarray(result)