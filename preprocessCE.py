import cupy as cp
from adaptFrequency import adaptFrequency

def preprocessCE(ce, ceSampleFrequency, requiredFrequency, expectedLength):

    # validate inputs
    if not isinstance(ce, cp.ndarray) or ce.ndim != 1:
        raise ValueError("'ce' must be a numeric vector")

    if not isinstance(ceSampleFrequency, (int, float, cp.ndarray)) or ceSampleFrequency <= 0:
        raise ValueError("'ceSampleFrequency' must be a positive scalar")

    if not isinstance(requiredFrequency, (int, float, cp.ndarray)) or requiredFrequency <= 0:
        raise ValueError("'requiredFrequency' must be a positive scalar")

    if not isinstance(expectedLength, (int, float, cp.ndarray)) or expectedLength <= 0 or expectedLength != int(expectedLength):
        raise ValueError("'expectedLength' must be a positive integer scalar")
    # measInfo.expectedAScanLength (the usual caller) is a float; cast once so
    # every slice bound / array-size use below gets a genuine int
    expectedLength = int(expectedLength)


    # adapt frequency of ce
    ce = adaptFrequency(ce, ceSampleFrequency, requiredFrequency)

    #remove offset
    ce -= cp.mean(ce)

    #normalization
    ce = ce / cp.max(cp.abs(ce))
    ce = ce / cp.sum(cp.abs(ce))

    if ce.shape[0] < expectedLength:
        # padding to size of ascan
        pad = cp.zeros(expectedLength - ce.shape[0], dtype=ce.dtype)
        ce = cp.concatenate([ce, pad])
    elif ce.shape[0] > expectedLength:
        # cropping to size of ascan
        ce = ce[:expectedLength]

    # extend further: pad with repeated last value up to expectedLength*2
    numNewElements = expectedLength * 2 - ce.shape[0]
    if numNewElements > 0:
        padVal = cp.full(numNewElements, ce[-1])
        ce = cp.concatenate([ce, padVal])

    ce = ce.ravel()

    return ce