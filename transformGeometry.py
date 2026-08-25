import cupy as cp
from types import SimpleNamespace
from rotateAndTranslate import rotateAndTranslate

def transformGeometry(TASElements, motorPos, rlList, rnList, slList, snList, transformationMatrices):
    
    # input validation
    if not isinstance(TASElements, (SimpleNamespace, list)):
        raise ValueError("'TASElements' must be a struct-like object")

    if not isinstance(motorPos, cp.ndarray) or motorPos.ndim != 1 or cp.any(motorPos <= 0):
        raise ValueError("'motorPos' must be a row vector of positive numbers")

    if not isinstance(rlList, cp.ndarray) or rlList.ndim != 1 or cp.any(rlList <= 0):
        raise ValueError("'rlList' must be a row vector of positive numbers")

    if not isinstance(rnList, cp.ndarray) or rnList.ndim != 1 or cp.any(rnList <= 0):
        raise ValueError("'rnList' must be a row vector of positive numbers")

    if not isinstance(slList, cp.ndarray) or slList.ndim != 1 or cp.any(slList <= 0):
        raise ValueError("'slList' must be a row vector of positive numbers")

    if not isinstance(snList, cp.ndarray) or snList.ndim != 1 or cp.any(snList <= 0):
        raise ValueError("'snList' must be a row vector of positive numbers")

    if not isinstance(transformationMatrices, cp.ndarray) or transformationMatrices.ndim != 3:
        raise ValueError("'transformationMatrices' must be a 3D numeric array")
    
    motorPos = cp.unique(motorPos)

    receiverNormals = cp.full((3, int(cp.max(rnList)), int(cp.max(rlList)), int(cp.max(motorPos))), cp.nan)
    receiverPositions = cp.full((3, int(cp.max(rnList)), int(cp.max(rlList)), int(cp.max(motorPos))), cp.nan)
    senderNormals = cp.full((3, int(cp.max(snList)), int(cp.max(slList)), int(cp.max(motorPos))), cp.nan)
    senderPositions = cp.full((3, int(cp.max(snList)), int(cp.max(slList)), int(cp.max(motorPos))), cp.nan)


    for mp in motorPos:
        mp = int(mp)
        mp_idx = mp - 1  # 1-based -> 0-based for array indexing
        
        # transform the geometry with the given stored transformation matrix
        transMat = cp.squeeze(transformationMatrices[:, :, mp_idx])
        transMatNormals = cp.linalg.inv(transMat).T
        
        # receiver position, normal calculation
        for rL in rlList:
            rL = int(rL)
            rL_idx = rL - 1
            for rN in rnList:
                rN = int(rN)
                rN_idx = rN - 1
                # [rN_idx:rN_idx+1, :] (not [rN_idx, :]) keeps the row 2D --
                # Python indexing with a plain int collapses to 1D, unlike
                # MATLAB's A(i,:), and rotateAndTranslate requires 2D input
                receiverNormals[:, rN_idx, rL_idx, mp_idx] = rotateAndTranslate(transMatNormals, TASElements[rL_idx].receiverNormals[rN_idx:rN_idx + 1, :]).ravel()
                receiverPositions[:, rN_idx, rL_idx, mp_idx] = rotateAndTranslate(transMat, TASElements[rL_idx].receiverPositions[rN_idx:rN_idx + 1, :]).ravel()
        
        # sender position, normal calculation
        for sL in slList:
            sL = int(sL)
            sL_idx = sL - 1
            for sN in snList:
                sN = int(sN)
                sN_idx = sN - 1
                senderNormals[:, sN_idx, sL_idx, mp_idx] = rotateAndTranslate(transMatNormals, TASElements[sL_idx].emitterNormals[sN_idx:sN_idx + 1, :]).ravel()
                senderPositions[:, sN_idx, sL_idx, mp_idx] = rotateAndTranslate(transMat, TASElements[sL_idx].emitterPositions[sN_idx:sN_idx + 1, :]).ravel()


    return senderNormals, receiverNormals, senderPositions, receiverPositions
