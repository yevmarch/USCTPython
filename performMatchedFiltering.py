import cupy as cp
from precalculateChannelList import precalculateChannelList

def performMatchedFiltering(AscanBlock, rlList, rnList, rlBlock, rnBlock, geom, expInfo, preComputes):
    channelList = precalculateChannelList(rlList, rnList, geom.headTable, expInfo, preComputes)

    rl_idx = rlBlock.flatten().astype(cp.int64) - 1  
    rn_idx = rnBlock.flatten().astype(cp.int64) - 1  
    # channelList[..., 0] is the channel index into matchedFilter's columns
    # (1 and 2 are mux/polarity, not used here)
    channelBlock = channelList[rl_idx, rn_idx, 0].astype(cp.int64)

    originalAscanLength = AscanBlock.shape[0]

    expInfo.matchedFilter = preComputes.matchedFilter

    FX = cp.fft.fft(AscanBlock, n=AscanBlock.shape[0]*2, axis=0)
    FH = expInfo.matchedFilter[:, channelBlock]
    AscanBlockMatchedFiltered = cp.real(cp.fft.ifft(
        cp.array(cp.real(FX) * cp.real(FH) + cp.imag(FX) * cp.imag(FH) +1j * (cp.imag(FX) * cp.real(FH) - cp.real(FX) * cp.imag(FH))), axis=0
    ))

    # trim to original length
    AscanBlockMatchedFiltered = AscanBlockMatchedFiltered[:originalAscanLength, :]

    # get rid of artifacts at the beginning and at the end of a signal
    AscanBlockMatchedFiltered[0:100, :] = 0
    AscanBlockMatchedFiltered[-101:, :] = 0

    return AscanBlockMatchedFiltered