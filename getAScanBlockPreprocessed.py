import cupy as cp
from types import SimpleNamespace
from getAScanBlock import getAScanBlock
from blockingGeometryInfos import blockingGeometryInfos
from preprocessAScanBlock import preprocessAScanBlock

def getAScanBlockPreprocessed(rootUniqueID, path, mp, sl, sn, rl, rn, geom, measInfo, params, applyFilter, transReco):
    # validate inputs
    if applyFilter not in (0, 1, True, False):
        raise ValueError("'applyFilter' must be a binary scalar (0/1 or True/False)")

    if transReco is not None:
        if not all(v in (0, 1, True, False) for v in cp.atleast_1d(transReco)):
            raise ValueError("'transReco' must be binary (0/1 or True/False)")

    # getAScanBlock requires 2D (1, N) row vectors, but callers commonly pass
    # a plain int (mp=1) or a 1D array (sl=cp.arange(...)) -- normalize here
    # rather than loosening getAScanBlock's own input contract
    mp = cp.atleast_2d(cp.asarray(mp))
    sl = cp.atleast_2d(cp.asarray(sl))
    sn = cp.atleast_2d(cp.asarray(sn))
    rl = cp.atleast_2d(cp.asarray(rl))
    rn = cp.atleast_2d(cp.asarray(rn))

    # get blocked AScan data
    AscanBlock, mpBlock, slBlock, snBlock, rlBlock, rnBlock, gainBlock = getAScanBlock(path, mp, sl ,sn, rl, rn, measInfo, rootUniqueID)


    # blocking accordingly the geometric information
    senderPositionBlock, senderNormalBlock, receiverPositionBlock, receiverNormalBlock = blockingGeometryInfos(geom, rnBlock, rlBlock, snBlock, slBlock, mpBlock)


    # filter data, define data which should be used
    # is unused, since applyFilter=false and transReco=0
    #if applyFilter:
    #    if transReco:
    #        usedData = filterTransmissionData(params.dataSelection, slBlock, snBlock, rlBlock, rnBlock, geom.sensData, senderNormalBlock, receiverNormalBlock)
    #    else:
    #        usedData = filterReflectionData(receiverPositionBlock, senderPositionBlock, senderNormalBlock, params.dataSelection)


    #    # remove unused data
    #    AscanBlock, mpBlock, slBlock, snBlock, rlBlock, rnBlock, senderPositionBlock, receiverPositionBlock, gainBlock, senderNormalBlock, receiverNormalBlock = ...
    #    removeDataFromArrays(usedData, AscanBlock, mpBlock, slBlock, snBlock, rlBlock, rnBlock, senderPositionBlock, receiverPositionBlock, gainBlock, senderNormalBlock, receiverNormalBlock)


    # A-Scan block preprocessing (formatting)
    if AscanBlock.size > 0:
        AscanBlockPreprocessed = preprocessAScanBlock(AscanBlock, measInfo, params.dataPreparation)
        AscanBlockPreprocessed = AscanBlockPreprocessed / gainBlock
    else:
        AscanBlockPreprocessed = AscanBlock


    return AscanBlockPreprocessed, mpBlock, slBlock, snBlock, rlBlock, rnBlock, senderPositionBlock, receiverPositionBlock, gainBlock, senderNormalBlock, receiverNormalBlock