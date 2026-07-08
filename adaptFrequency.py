import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline

def adaptFrequency(ce, ceSampleFrequency, requiredFrequency):

    # antialiasing filtering and downsampling if necessary downsampling
    if requiredFrequency < ceSampleFrequency:
        f2 = cp.linspace(0, ceSampleFrequency, len(ce))
        df = cp.fft.fft(ce)
        df[f2 > requiredFrequency] = 0
        ce = cp.fft.irfft(df[:len(ce) // 2 + 1], n=len(ce))

    x = np.arange(1, len(ce) + 1)
    xq = np.arange(1, len(ce) + 1, ceSampleFrequency / requiredFrequency)
    ce = cp.asarray(CubicSpline(x, cp.asnumpy(ce))(xq))

    return ce