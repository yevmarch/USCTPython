import cupy as cp
from types import SimpleNamespace

def reviseMatchedFilter(matchedFilter, removeOutliers):

    if removeOutliers:
        mFTime = cp.fft.ifft(matchedFilter, axis=0)
        normSTD = cp.std(cp.abs(mFTime), axis=0, ddof=1)
        normSTD[cp.isnan(normSTD)] = 0
        
        sortIndex = cp.argsort(normSTD)
        sortSTD = normSTD[sortIndex]
        
        t = round(0.4 * mFTime.shape[1])
        if t <= 0:
            t = 1  # prevent zero indexing
        
        threshold = sortSTD[t - 1]  # MATLAB 1-based -> 0-based
        maxAbsPerCol = cp.max(cp.abs(mFTime), axis=0)
        globalMax = cp.max(cp.abs(mFTime))
        
        indexe = cp.where((normSTD < threshold) | (maxAbsPerCol < 0.1 * globalMax))[0]
        
        maxMF = mFTime[:, sortIndex[-1]]  # set to MF (highest-STD column)
        
        for ces in indexe:
            mFTime[:, ces] = maxMF
        
        matchedFilter = cp.fft.fft(mFTime, axis=0)


    # calculate noise level
    highNoiseScore = cp.mean(cp.abs(matchedFilter) * cp.std(cp.abs(matchedFilter), axis=0, ddof=1))

    # find median ce
    index = cp.argmin(cp.abs(highNoiseScore - cp.median(highNoiseScore)))

    #find entries with too large deviations
    sumDiff = cp.sum(cp.abs(matchedFilter - matchedFilter[:, index:index+1]), axis=0)
    indexe = (sumDiff > cp.mean(sumDiff) + 2.596 * cp.std(sumDiff, axis=0, ddof=1))

    # replace them with the median one 
    matchedFilter[:, indexe] = cp.tile(matchedFilter[:, index:index+1], (1, int(cp.sum(indexe))))

    return matchedFilter