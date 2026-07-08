import cupy as cp
from USCTTAS2DAQChannels import USCTYTAS2DAQChannels

def precalculateChannelList(rlList, rnList, headTable, expInfo, preComputes):
    channelList = cp.zeros((rlList.size, rnList.size))
    for rl in rlList:
        for rn in rnList:
            channelList[rl, rn, :] = USCTYTAS2DAQChannels(headTable, rl, rn, expInfo, preComputes)

    return channelList