import cupy as cp
from types import SimpleNamespace
from reviseMatchedFilter import reviseMatchedFilter
from xcorr import _xcorr
from writeReconstructionLog import writeReconstructionLog


def estimateOffset(flags, info, ce, matchedFilter):

    offset = 0.0

    if ce.ceAvailable:
        CE = ce.CERef
        CEOffset = ce.CERefOffset
    else:
        raise ValueError("CE is not vailable")

    if ce.measuredCEavailable:
        if ce.measuredCEused:
            CEMeasured = cp.fft.ifft(matchedFilter, axis=0)
        else:
            CEMeasured = cp.fft.ifft(reviseMatchedFilter(cp.fft.fft(ce.CE, axis=0), 1), axis=0)
    else:
        raise ValueError("measuredCE is not available")
    

    # include EOffset from configFile
    offset = offset + flags.dataPreparation.offsetElectronic

    # estimate offset in CE in case CEMeasured is not used
    if not ce.measuredCEused:
        if ce.measuredCEavailable and ce.ceAvailable and ce.measuredCEusedForCEcompensation:
            offsetsCEMeasured = cp.zeros(CEMeasured.shape[1])
            for ces in range(ce.CE.shape[1]):
                corrCE = _xcorr(CEMeasured[:, ces], CE)
                offsetsCEMeasured[ces] = cp.argmax(corrCE)
                offsetsCEMeasured[ces] = offsetsCEMeasured[ces] - CE.shape[0]
            
            offset = offset + float(cp.median(offsetsCEMeasured)) / flags.dataPreparation.aScanReconstructionFrequency
            
            if info.Hardware.lower() == 'usct3dv2':
                offset -= CEOffset
                CEOffset = 0
        
        else:
            # trying to estimate the offset from the meta data.
            writeReconstructionLog('Using CE, but compensation of delay could not be done since CEMeasured is not available', 3)
            writeReconstructionLog('Trying to estimate the delay from the measurement meta data', 2)

            try:
                filterDisabled = info.MetaData.FilterBypass
                filterDisableDetected = 1
            except:
                filterDisabled = 0
                filterDisableDetected = 0

            if filterDisableDetected:
                if filterDisabled:
                    additionalOffset = ce.offsetFilterDisabled
                else:
                    additionalOffset = ce.offsetFilterEnabled
                offset += additionalOffset
            
            else:
                writeReconstructionLog('Estimation of delay could not be done since DACDelay in measurement meta data is not available. Assuming zero additional offset. Please check the config file if you have selected an appropriate electronic offset value!', 3)


    # init offset
    if info.Hardware.lower() == "usct3dv2" and info.Bandpassundersampling == 1:
        Digitalfilterdelay = -0.8e-6
    elif info.Hardware.lower() == "usct3dv2":
        Digitalfilterdelay = -(4.67e-6 + 0.8e-6)
    elif info.Hardware.lower() == "usct3dv3":
        Digitalfilterdelay = 0
    else:
        Digitalfilterdelay = 0

    
    if not ce.measuredCEused:
        offset += Digitalfilterdelay
        if CEOffset is not None:
            offset += CEOffset

    if hasattr(info, 'EOffset'):
        offset -= info.EOffset


    return offset