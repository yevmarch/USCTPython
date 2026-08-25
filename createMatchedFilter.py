import cupy as cp
from types import SimpleNamespace
from reviseMatchedFilter import reviseMatchedFilter

def createMatchedFilter(ce, measuredCEused, params, removeOutliersFromCEMeasured, hardwareVersion):
    # input validation
    if not isinstance(ce, cp.ndarray) or ce.ndim != 2:
        raise ValueError("'ce' must be a 2D numeric array")

    if measuredCEused not in (0, 1, True, False):
        raise ValueError("'measuredCEused' must be a binary scalar (0/1 or True/False)")

    if not isinstance(params, SimpleNamespace):
        raise ValueError("'params' must be a struct-like object")

    if removeOutliersFromCEMeasured not in (0, 1, True, False):
        raise ValueError("'removeOutliersFromCEMeasured' must be a binary scalar (0/1 or True/False)")
    
    matchedFilter = cp.fft.fft(ce, axis=0)

    if measuredCEused and params.findDefects == 1:
        matchedFilter = reviseMatchedFilter(matchedFilter, removeOutliersFromCEMeasured)

    if measuredCEused:
        if hardwareVersion.lower() == "usct3dv3":
            mFTime = cp.fft.ifft(-matchedFilter, axis=0)
        else:
            mFTime = cp.fft.ifft(matchedFilter, axis=0)

        # normalize
        mFTime = mFTime / cp.max(cp.abs(mFTime), axis=0)
        mFTime = mFTime / cp.sum(cp.abs(mFTime), axis=0)
        matchedFilter = cp.fft.fft(mFTime, axis=0)

    return matchedFilter

