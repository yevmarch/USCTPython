import cupy as cp
from types import SimpleNamespace
from convertfp16tofloat import convertfp16tofloat
from reconstructBandpasssubsampling import reconstructBandpasssubsampling

def preprocessAScanBlock(AScans, measInfo, params):

    # input validation
    if not isinstance(AScans, cp.ndarray) or AScans.ndim != 2:
        raise ValueError("'AScans' must be a 2D numeric array")
    
    if not isinstance(measInfo, SimpleNamespace):
         raise ValueError("'measInfo' must be a struct-like object")
    
    if not isinstance(params, SimpleNamespace):
         raise ValueError("'params' must be a struct-like object")

    # convert float 16 format
    if hasattr(measInfo, 'AScanDataType') and measInfo.AScanDataType.lower() == "float16":
         AScans = convertfp16tofloat(AScans)

    # remove nans
    AScans[cp.isnan(AScans)] = 0

    # ensure float64 format 
    AScans = AScans.astype(cp.float64)

    # reconstruct bandpass undersampling
    if hasattr(measInfo, 'Bandpassundersampling') and measInfo.Bandpassundersampling == 1:
         AScans = reconstructBandpasssubsampling(AScans, params.aScanReconstructionFrequency, measInfo.SampleRate)

    # check if number samples fit to expected ones
    if AScans.shape[0] != measInfo.expectedAScanLength:
         raise ValueError(
            "preprocessAScanBlock:InvalidSize - "
            "Samples of AScan data did not correspond to expected AScan length"
         )
    
    return AScans