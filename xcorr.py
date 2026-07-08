import cupy as cp

def _xcorr(a, b):
    """Equivalent of MATLAB's xcorr(a, b) - cross-correlation across all lags."""
    n = len(a) + len(b) - 1
    nfft = 1
    while nfft < n:
        nfft *= 2
    A = cp.fft.fft(a, nfft)
    B = cp.fft.fft(b, nfft)
    corr = cp.fft.ifft(A * cp.conj(B)).real
    corr = cp.concatenate([corr[-(len(b)-1):], corr[:len(a)]])
    return corr