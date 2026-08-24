import cupy as cp

def correctTASTemperatures(tempArray, refTemps):
    if not isinstance(tempArray, cp.ndarray) or tempArray.ndim != 2:
        raise ValueError("'tempArray' must be a 2D numeric array")

    if not isinstance(refTemps, cp.ndarray) or refTemps.shape[1] != tempArray.shape[1]:
        raise ValueError(f"'refTemps' must have {tempArray.shape[1]} columns")

    # mean of non-NaN reference temperatures (per column)
    meanTempCali = cp.mean(
        refTemps[~cp.isnan(refTemps)].reshape(refTemps.shape), axis=0
    )

    for aperturePos in range(tempArray.shape[1]):
        # experimental linear fit
        temps = 1.1294 * tempArray[:, aperturePos] - 1.1715
        temps = temps + meanTempCali[aperturePos] - cp.mean(temps[~cp.isnan(temps)])
        temps[cp.isnan(temps)] = meanTempCali[aperturePos]  # fill
        tempArray[:, aperturePos] = temps

    return tempArray
