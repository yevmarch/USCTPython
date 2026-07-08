import cupy as cp

def reconstructBandpasssubsampling(Data, AScanReconstructionFreq, SampleRate):
    downsamplingfactor = AScanReconstructionFreq / SampleRate
    minimalExpectedAScanLength = 48

    expectedAScanlengthDS = int(cp.ceil(minimalExpectedAScanLength / downsamplingfactor))

    """greater ascan than 3000!"""
    expectedAScanLength = int(cp.max(expectedAScanlengthDS * downsamplingfactor, len(Data.shape[0])))
    expectedAScanlengthDS = int(cp.ceil(expectedAScanlengthDS / downsamplingfactor))

    s1 = AScanReconstructionFreq
    f1 = cp.linspace(0, s1, expectedAScanLength)
        #         f1_2 = f1[0:1+len(f1)//2]
        #         s2 = SampleRate
        #         f2 = cp.arange(0, s2 + s2/(Data.shape[0]-1), s2/(Data.shape[0]-1))  # if more precise assignment is required...
        #         f3 = f2 + SampleRate
        #         f3_2 = f3[0:1+len(f3)//2]
    Data2 = cp.zeros((expectedAScanLength, Data.shape[1]), dtype=cp.float64)

    datafft = cp.fft.fftshift(cp.fft.fft(Data, n=expectedAScanLength, axis=0), axes=0)
    # mirror boundary frequency
    datafft[expectedAScanlengthDS // 2] = 0

    SR = SampleRate

    idx_l = cp.where(f1 > SampleRate/2 & f1 <= SR) # resolution of f1 as tolerance
    idx_r = cp.where(f1 > AScanReconstructionFreq-SR & f1 <= AScanReconstructionFreq-(SR/2))

    if len(idx_l) == 0 or len(idx_r) == 0:
        print('downsampled data, you have to set AScanReconstructionFreq higher')
        return

    if datafft.shape[0] / 2 != len(idx_r):
        print('error: Fourier resolution (->length) needs to be fixed by padding')
        return
    
    for j in range(Data.shape[0]):
        Data2[idx_l, j] = datafft[:datafft.shape[0] // 2, j]
        Data2[idx_r + 1, j] = cp.flipud(cp.conj(datafft[:datafft.shape[0] // 2, j]))

    # in old version without scaling to downsamplingfactor
    # Data2 = cp.fft.ifft(Data2, axis=0)

    return Data2