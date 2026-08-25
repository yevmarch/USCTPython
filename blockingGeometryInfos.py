import cupy as cp
from types import SimpleNamespace

def blockingGeometryInfos(geom, rnBlock, rlBlock, snBlock, slBlock, mpBlock):
    # input validation
    if not isinstance(geom, SimpleNamespace):
        raise ValueError("'geom' must be a struct-like object")

    if not isinstance(rnBlock, cp.ndarray) or rnBlock.ndim != 2 or rnBlock.shape[0] != 1 or not cp.all(rnBlock == rnBlock.astype(cp.int32)) or not cp.all(rnBlock > 0):
        raise ValueError("'rnBlock' must be a row vector of positive integers")

    n = rnBlock.size

    if not isinstance(rlBlock, cp.ndarray) or rlBlock.ndim != 2 or rlBlock.shape[0] != 1 or rlBlock.size != n or not cp.all(rlBlock == rlBlock.astype(cp.int32)) or not cp.all(rlBlock > 0):
        raise ValueError(f"'rlBlock' must be a row vector of {n} positive integers")

    if not isinstance(snBlock, cp.ndarray) or snBlock.ndim != 2 or snBlock.shape[0] != 1 or snBlock.size != n or not cp.all(snBlock == snBlock.astype(cp.int32)) or not cp.all(snBlock > 0):
        raise ValueError(f"'snBlock' must be a row vector of {n} positive integers")

    if not isinstance(slBlock, cp.ndarray) or slBlock.ndim != 2 or slBlock.shape[0] != 1 or slBlock.size != n or not cp.all(slBlock == slBlock.astype(cp.int32)) or not cp.all(slBlock > 0):
        raise ValueError(f"'slBlock' must be a row vector of {n} positive integers")

    if not isinstance(mpBlock, cp.ndarray) or mpBlock.ndim != 2 or mpBlock.shape[0] != 1 or mpBlock.size != n or not cp.all(mpBlock == mpBlock.astype(cp.int32)) or not cp.all(mpBlock > 0):
        raise ValueError(f"'mpBlock' must be a row vector of {n} positive integers")
    
    if geom.receiverPositions.shape != geom.receiverNormals.shape:
        raise ValueError(
            "blockingGeometryInfos:InvalidSize - "
            "It is assumed that the geometric information, receiver position and normals, are of the same size."
        )
    
    if geom.senderPositions.shape != geom.senderPositions.shape:
        raise ValueError(
            "blockingGeometryInfos:InvalidSize - "
            "It is assumed that the geometric information, sender position and normals, are of the same size."
        )
    

    # number of data
    numData = rnBlock.shape[1]
    arrayOfDataSize = cp.ones((1, numData), dtype=rnBlock.dtype)

    # blocking receiver infos: normals and positions
    dim = geom.receiverPositions.shape[0]
    receiverPositionBlock = cp.zeros((dim, numData))
    receiverNormalBlock = cp.zeros((dim, numData))

    for i in range(1, dim + 1):
    # sub2ind equivalent: directly index 4D array with per-dimension index arrays
    # arrayOfDataSize*i gives a row of repeated value i (1-based dimension index)
        dim_idx = (arrayOfDataSize * i).astype(cp.int64) - 1      # convert to 0-based
        rn_idx  = rnBlock.flatten().astype(cp.int64) - 1          # convert to 0-based
        rl_idx  = rlBlock.flatten().astype(cp.int64) - 1          # convert to 0-based
        mp_idx  = mpBlock.flatten().astype(cp.int64) - 1          # convert to 0-based

        receiverPositionBlock[i - 1, :] = geom.receiverPositions[dim_idx, rn_idx, rl_idx, mp_idx]
        receiverNormalBlock[i - 1, :]   = geom.receiverNormals[dim_idx, rn_idx, rl_idx, mp_idx]

    # blocking emitter infos: normals and positions
    dim = geom.senderPositions.shape[0]
    senderPositionBlock = cp.zeros((dim, numData))
    senderNormalBlock = cp.zeros((dim, numData))

    for i in range(1, dim + 1):
    # sub2ind equivalent: directly index 4D array with per-dimension index arrays
    # arrayOfDataSize*i gives a row of repeated value i (1-based dimension index)
        dim_idx = (arrayOfDataSize * i).astype(cp.int64) - 1      # convert to 0-based
        sn_idx  = snBlock.flatten().astype(cp.int64) - 1          # convert to 0-based
        sl_idx  = slBlock.flatten().astype(cp.int64) - 1          # convert to 0-based
        mp_idx  = mpBlock.flatten().astype(cp.int64) - 1          # convert to 0-based

        senderPositionBlock[i - 1, :] = geom.senderPositions[dim_idx, sn_idx, sl_idx, mp_idx]
        senderNormalBlock[i - 1, :]   = geom.senderNormals[dim_idx, sn_idx, sl_idx, mp_idx]

    
    return senderPositionBlock, senderNormalBlock, receiverPositionBlock, receiverNormalBlock