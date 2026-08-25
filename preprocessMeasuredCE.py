import cupy as cp
from types import SimpleNamespace
from convertfp16tofloat import convertfp16tofloat
from reconstructBandpasssubsampling import reconstructBandpasssubsampling

def preprocessMeasuredCE(ce, measInfo, reconstructionFreq, removeDCOffset):

    # validate inputs
    if not isinstance(ce, SimpleNamespace):
        raise ValueError("'ce' must be a struct-like object")

    if not isinstance(measInfo, SimpleNamespace):
        raise ValueError("'measInfo' must be a struct-like object")

    # reconstructionFreq is typically a native float (e.g. measInfo.SampleRate
    # after loadFileToStruct's normalization) -- cp.size() requires an actual
    # cupy array and rejects plain Python numbers outright, so only call it
    # once isinstance has confirmed it isn't one
    _reconstructionFreqOk = isinstance(reconstructionFreq, (int, float)) or (
        isinstance(reconstructionFreq, cp.ndarray) and cp.size(reconstructionFreq) == 1
    )
    if not _reconstructionFreqOk:
        raise ValueError("'reconstructionFreq' must be a numeric scalar")

    if removeDCOffset not in (0, 1):
        raise ValueError("'removeDCOffset' must be a binary scalar (0 or 1)")
    
    # get ce to right format
    if hasattr(measInfo, 'AScanDatatype') and measInfo.AScanDatatype == 'float16':
        ce.CEMeasured = convertfp16tofloat(ce.CEMeasured)

    ceOut = cp.asarray(ce.CEMeasured, dtype=cp.float64)

    # decompress if necessary
    if measInfo.Bandpassundersampling == 1:
        # expand to measInfo.NumberSamples rows, padding with zeros
        # (measInfo.NumberSamples is a float; cast since it's used as an
        # array-size argument and a slice bound below, both of which require ints)
        numberSamples = int(measInfo.NumberSamples)
        if ceOut.shape[0] < numberSamples:
            pad = cp.zeros((numberSamples - ceOut.shape[0], ceOut.shape[1]), dtype=ceOut.dtype)
            ceOut = cp.vstack([ceOut, pad])
        elif ceOut.shape[0] > numberSamples:
            ceOut = ceOut[:numberSamples, :]
        
        ceOut = reconstructBandpasssubsampling(ceOut, reconstructionFreq, ce.CE_SF)

    
    # pad to double the length of an A-Scan, filling the new rows with a
    # repeated copy of the current last row.
    # (measInfo.NumberSamples is a float, so cast the target length to int;
    # the original `ceOut[indInsert, :] = ...` relied on MATLAB's auto-growing
    # array assignment -- numpy/cupy don't support that, assigning beyond the
    # array's current bounds just raises IndexError -- so the padding has to
    # be built and concatenated instead)
    targetLength = int(measInfo.NumberSamples * 2)
    if targetLength > ceOut.shape[0]:
        numInsert = targetLength - ceOut.shape[0]
        pad = cp.tile(ceOut[ceOut.shape[0] - 1, :], (numInsert, 1))
        ceOut = cp.vstack([ceOut, pad])

    #offset removal
    if removeDCOffset:
        ceOut -= cp.mean(ceOut)

    return ceOut

    
            