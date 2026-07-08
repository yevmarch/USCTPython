import cupy as cp

def checkForCommonListEntries(*varargin):
    usedData = cp.full(cp.asarray(varargin[0]).shape, True)

    for i in range(0, len(varargin), 2):
        usedData = usedData & cp.isin(cp.asarray(varargin[i]), cp.asarray(varargin[i + 1]))

    return usedData