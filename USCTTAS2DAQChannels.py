import cupy as cp
from types import SimpleNamespace

def USCTYTAS2DAQChannels(headTable, TASnr, Receivernr, expInfo, preComputes):

    if expInfo.Hardware.lower() == "usct3dv2":
        """ init and HW defines """

        HW = SimpleNamespace()
        HW.FLTnumber = 20
        HW.FLTChannelnumber = 24
        HW.Channels = cp.arange(1, HW.FLTnumber * HW.FLTChannelnumber + 1)
        HW.recElementIdx2RecChannelMUX = cp.array([1, 3], [1, 2], [1, 1], [2, 3], [2, 2], [2, 1], [3, 3], [3, 2], [3, 1]) # linear idx in geometryfile AND data to the HW and MUX
        HW.emitElementIdx2Emitter = cp.array([2, 1, 4, 3]) # linear idx in geometryfile AND data to the HW
        HW.FLTChannelPolarity = cp.array([-1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1]) # WORKAROUND FOR RJ45 socket signal inversion

        FLTs = 1 + cp.floor((HW.Channels - 1) / HW.FLTChannelnumber).astype(cp.int32)
        FLTChannels = HW.Channels % HW.FLTChannelnumber
        FLTChannels[FLTChannels == 0] = HW.FLTChannelnumber

        TASChannel = HW.recElementIdx2RecChannelMUX[Receivernr - 1, 0]  # 1-based -> 0-based
        mux = HW.recElementIdx2RecChannelMUX[Receivernr - 1, 1]         # 1-based -> 0-based

        tidx = cp.where((headTable[:, 0] == TASnr) & (headTable[:, 3] == TASChannel))[0]  # 1-based cols -> 0-based

        FLT = headTable[tidx, 1]        # column 2 -> index 1
        FLTChannel = headTable[tidx, 2]  # column 3 -> index 2

        channel = cp.where((FLTs == FLT) & (FLTChannels == FLTChannel))[0]

        polarity = HW.FLTChannelPolarity[FLTChannel - 1]

    elif expInfo.Hardware.lower() == "usct3dv3" or expInfo.Hardware.lower() == "usct3dv3_simulated":
        if preComputes.measuredCEused:
            channel = cp.where(preComputes.measuredCE_TASIndices == TASnr & preComputes.measuredCE_receiverIndices == Receivernr)

            if channel is None or len(channel) == 0:
                channel = 1

        mux = 1
        polarity = 1

    return channel, mux, polarity