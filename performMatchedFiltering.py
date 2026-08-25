import cupy as cp
from precalculateChannelList import precalculateChannelList
from estimateBatchSize import estimateBatchSize

def performMatchedFiltering(AscanBlock, rlList, rnList, rlBlock, rnBlock, geom, expInfo, preComputes):
    channelList = precalculateChannelList(rlList, rnList, geom.headTable, expInfo, preComputes)

    rl_idx = rlBlock.flatten().astype(cp.int64) - 1
    rn_idx = rnBlock.flatten().astype(cp.int64) - 1
    # channelList[..., 0] is the channel index into matchedFilter's columns
    # (1 and 2 are mux/polarity, not used here)
    channelBlock = channelList[rl_idx, rn_idx, 0].astype(cp.int64)

    originalAscanLength = AscanBlock.shape[0]
    fftLength = originalAscanLength * 2
    numAScans = AscanBlock.shape[1]

    expInfo.matchedFilter = preComputes.matchedFilter

    AscanBlockMatchedFiltered = cp.zeros((originalAscanLength, numAScans), dtype=cp.float64)

    # doing this in one shot needs several fftLength x numAScans complex128
    # arrays alive at once (FX, the fancy-indexed FH, the intermediate
    # complex product, the ifft result) -- at full scale (tens of thousands
    # of A-scans) that easily exceeds GPU memory. Batch it, the same way
    # reconstructSAFTimage.py already does for its own per-voxel work.
    batchSize = estimateBatchSize((fftLength,), dtype=cp.complex128)

    for start in range(0, numAScans, batchSize):
        end = min(start + batchSize, numAScans)

        FX = cp.fft.fft(AscanBlock[:, start:end], n=fftLength, axis=0)
        FH = expInfo.matchedFilter[:, channelBlock[start:end]]
        filtered = cp.real(cp.fft.ifft(
            cp.array(cp.real(FX) * cp.real(FH) + cp.imag(FX) * cp.imag(FH) + 1j * (cp.imag(FX) * cp.real(FH) - cp.real(FX) * cp.imag(FH))), axis=0
        ))

        # trim to original length
        AscanBlockMatchedFiltered[:, start:end] = filtered[:originalAscanLength, :]

        del FX, FH, filtered
        cp.get_default_memory_pool().free_all_blocks()

    # get rid of artifacts at the beginning and at the end of a signal
    AscanBlockMatchedFiltered[0:100, :] = 0
    AscanBlockMatchedFiltered[-101:, :] = 0

    return AscanBlockMatchedFiltered