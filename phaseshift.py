import cupy as cp
import numpy as np

def phaseshift(CE, delay, fs):
    delay = -delay

    trans = False
    if CE.ndim >= 2 and CE.shape[-1] == 1:
        CE = CE.ravel()
        trans = True

    N = len(CE)
    f = cp.linspace(0, fs, N)

    if N % 2 == 1:  # odd
        f = cp.roll(cp.fft.fftshift(f - fs / 2), 1)
    else:  # even
        f[N-1:N//2:-1] = -f[1:N//2]

    d = 2 * np.pi * delay * f   # = 2π·delay / wavel, since wavel=1/f
    ps = cp.exp(1j * d)

    if N % 2 == 0:              # fs/2 fix
        ps[N // 2] = 1.0

    CE = cp.real(cp.fft.ifft(cp.fft.fft(CE) * ps))

    if trans:
        CE = CE.reshape(-1, 1)

    return CE