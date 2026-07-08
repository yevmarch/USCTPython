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

    if not isinstance(reconstructionFreq, (int, float, cp.ndarray)) or cp.size(reconstructionFreq) != 1:
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
        if ceOut.shape[0] < measInfo.NumberSamples:
            pad = cp.zeros((measInfo.NumberSamples - ceOut.shape[0], ceOut.shape[1]), dtype=ceOut.dtype)
            ceOut = cp.vstack([ceOut, pad])
        elif ceOut.shape[0] > measInfo.NumberSamples:
            ceOut = ceOut[:measInfo.NumberSamples, :]
        
        ceOut = reconstructBandpasssubsampling(ceOut, reconstructionFreq, ce.CE_SF)

    
    # pad to double the length of an A-Scan 
    indInsert = cp.arange(ceOut.shape[0], measInfo.NumberSamples * 2)
    ceOut[indInsert, :] = cp.tile(ceOut[ceOut.shape[0], :], (len(indInsert), 1))

    #offset removal
    if removeDCOffset:
        ceOut -= cp.mean(ceOut)

    return ceOut

    
            