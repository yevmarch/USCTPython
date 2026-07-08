import cupy as cp

def hilbert(x, axis=0):
    N = x.shape[axis]
    Xf = cp.fft.fft(x, axis=axis)
    
    h = cp.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2 # double the amplitude of positive frequencies
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2 # double the amplitude of positive frequencies
    
    # reshape h for broadcasting along the correct axis
    shape = [1] * x.ndim
    shape[axis] = N
    h = h.reshape(shape)
    
    return cp.fft.ifft(Xf * h, axis=axis)