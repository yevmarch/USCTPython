import cupy as cp
from USCTTAS2DAQChannels import USCTYTAS2DAQChannels

def precalculateChannelList(rlList, rnList, headTable, expInfo, preComputes):
    # 3rd axis holds (channel, mux, polarity); rl/rn are 1-based TAS/receiver
    # numbers used directly (minus 1) as indices both here and at the
    # consumption site (performMatchedFiltering.py: channelList[rl_idx,
    # rn_idx]) -- so this must be sized by the max possible value, not by
    # rlList/rnList.size, which undercounts whenever the list is a sparse
    # sample (e.g. rn=[1,4,7,10,13,16], size 6 but max value 16)
    channelList = cp.zeros((int(cp.max(rlList)), int(cp.max(rnList)), 3))
    for rl in rlList:
        for rn in rnList:
            channel, mux, polarity = USCTYTAS2DAQChannels(headTable, rl, rn, expInfo, preComputes)
            if isinstance(channel, cp.ndarray):
                channel = int(channel[0]) if channel.size > 0 else cp.nan
            channelList[rl - 1, rn - 1, :] = channel, mux, polarity

    return channelList