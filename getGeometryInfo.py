import cupy as cp
from types import SimpleNamespace
from loadGeometry import loadGeometry
from loadChannelInfos import loadChannelInfos
from loadSensitivity import loadSensitivity
from transformGeometry import transformGeometry

def getGeometryInfo(files, motorPos, motorPosRef, rlList, rnList, slList, snList, transformationMatrices, transformationMatricesRef):

    geom = SimpleNamespace()
    geom.info = SimpleNamespace()

    # validate inputs
    if not isinstance(files, SimpleNamespace):
        raise ValueError("'files' must be a struct-like object")

    if not isinstance(motorPos, cp.ndarray) or motorPos.ndim != 1 or motorPos.size == 0 or cp.any(motorPos <= 0):
        raise ValueError("'motorPos' must be a non-empty row vector of positive numbers")

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

    if not (motorPosRef is None or motorPosRef.size == 0) or not (transformationMatricesRef is None or transformationMatricesRef.size == 0):
        if not isinstance(motorPosRef, cp.ndarray) or motorPosRef.ndim != 1 or cp.any(motorPosRef <= 0):
            raise ValueError("'motorPosRef' must be a row vector of positive numbers")
        if not isinstance(transformationMatricesRef, cp.ndarray) or transformationMatricesRef.ndim != 3:
            raise ValueError("'transformationMatricesRef' must be a 3D numeric array")
        

    # load geometry, setting informations
    geomInitial = loadGeometry(files.geometry)
    geom.headTable = loadChannelInfos(files.headTable)

    # load transducer sensitivity information
    geom.sensChar = loadSensitivity(files.transducerAngleCharacteristic)
    
    # transform according to measurement
    geom.senderNormals, geom.receiverNormals, geom.senderPositions, geom.receiverPositions = transformGeometry(geomInitial, motorPos, rlList, rnList, slList, snList, transformationMatrices)
    if not (motorPosRef is None or motorPosRef.size == 0) or not (transformationMatricesRef is None or transformationMatricesRef.size == 0):
        # only if data for transmission Reco given
        _, _, geom.senderRefPositions, geom.receiverRefPositions = transformGeometry(geomInitial, motorPosRef, rlList, rnList, slList, snList, transformationMatricesRef)



    # calculate additional geometry information

    # calculate number of elements
    geom.info.maxSL = len(geomInitial)
    geom.info.maxSN = geomInitial[0].emitterPositions.shape[0]
    geom.info.maxRL = len(geomInitial)
    geom.info.maxRN = geomInitial[0].receiverPositions.shape[0]
    geom.info.numTAS = max(geom.info.maxSL, geom.info.maxRL)

    #calculate min positions
    geom.info.minEmitter = cp.array([
        cp.min(geom.senderPositions[0, :, :, :]),
        cp.min(geom.senderPositions[1, :, :, :]),
        cp.min(geom.senderPositions[2, :, :, :])
    ]).reshape(3, 1)

    geom.info.minReceiver = cp.array([
        cp.min(geom.receiverPositions[0, :, :, :]),
        cp.min(geom.receiverPositions[1, :, :, :]),
        cp.min(geom.receiverPositions[2, :, :, :])
    ]).reshape(3, 1)

    geom.info.minSize = cp.minimum(geom.info.minEmitter, geom.info.minReceiver)

    #calculate max positions
    geom.info.maxEmitter = cp.array([
        cp.max(geom.senderPositions[0, :, :, :]),
        cp.max(geom.senderPositions[1, :, :, :]),
        cp.max(geom.senderPositions[2, :, :, :])
    ]).reshape(3, 1)

    geom.info.maxReceiver = cp.array([
        cp.max(geom.receiverPositions[0, :, :, :]),
        cp.max(geom.receiverPositions[1, :, :, :]),
        cp.max(geom.receiverPositions[2, :, :, :])
    ]).reshape(3, 1)

    geom.info.maxSize = cp.maximum(geom.info.maxEmitter, geom.info.maxReceiver)


    return geom
    
