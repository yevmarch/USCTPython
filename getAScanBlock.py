import cupy as cp
from types import SimpleNamespace
from loadAscanData import loadAscanData
from checkForCommonListEntries import checkForCommonListEntries
from writeReconstructionLog import writeReconstructionLog
from removeDataFromArrays import removeDataFromArrays
import warnings

def getAScanBlock(path, mp, sl, sn, rl ,rn, measInfo, rootUniqueID):
    
    # validate inputs
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if not isinstance(mp, cp.ndarray) or mp.ndim != 2 or mp.shape[0] != 1:
        raise ValueError("'mp' must be a row vector")

    if not isinstance(sl, cp.ndarray) or sl.ndim != 2 or sl.shape[0] != 1:
        raise ValueError("'sl' must be a row vector")

    if not isinstance(sn, cp.ndarray) or sn.ndim != 2 or sn.shape[0] != 1:
        raise ValueError("'sn' must be a row vector")

    if not isinstance(rl, cp.ndarray) or rl.ndim != 2 or rl.shape[0] != 1:
        raise ValueError("'rl' must be a row vector")

    if not isinstance(rn, cp.ndarray) or rn.ndim != 2 or rn.shape[0] != 1:
        raise ValueError("'rn' must be a row vector")

    if not isinstance(measInfo, SimpleNamespace):
        raise ValueError("'measInfo' must be a struct-like object")

    if not isinstance(rootUniqueID, str) or not rootUniqueID:
        raise ValueError("'rootUniqueID' must be a non-empty string")

    # initialize variables
    numScans = mp.shape[1] * sl.shape[1] * sn.shape[1] * rl.shape[1] * rn.shape[1]  # max number of scans expected due to input
    num = 0  # number of block entries

    if hasattr(measInfo, 'AScanDatatype') and measInfo.AScanDatatype == 'float16':
        dataType = cp.int16
    else:
        dataType = cp.float64

    AscanBlock = cp.zeros((measInfo.NumberSamples, numScans), dtype=dataType)
    slBlock = cp.zeros((1, numScans), dtype=cp.int16)
    snBlock = cp.zeros((1, numScans), dtype=cp.int16)
    rlBlock = cp.zeros((1, numScans), dtype=cp.int16)
    rnBlock = cp.zeros((1, numScans), dtype=cp.int16)
    mpBlock = cp.zeros((1, numScans), dtype=cp.int16)
    gainBlock = cp.ones((1, numScans))

    for mpE in mp.flatten():
        for slE in sl.flatten():
            for snE in sn.flatten():
                try:
                    # load data
                    dataBlock = loadAscanData(path, slE, snE, mpE, measInfo.Hardware, rootUniqueID)
                    
                    # check for required receiver data
                    usedData = checkForCommonListEntries(dataBlock.TASIndices, rl, dataBlock.receiverIndices, rn)
                    
                    # calculations for block writings
                    sizeDataBlock = int(cp.sum(usedData))
                    blockIdxs = slice(num, sizeDataBlock + num)
                    
                    # write required data to blocks
                    mpBlock[0, blockIdxs] = mpE
                    slBlock[0, blockIdxs] = slE
                    snBlock[0, blockIdxs] = snE
                    rlBlock[0, blockIdxs] = dataBlock.TASIndices[usedData]
                    rnBlock[0, blockIdxs] = dataBlock.receiverIndices[usedData]
                    AscanBlock[:, blockIdxs] = dataBlock.AScans[:, usedData]
                    
                    gainBlock[0, blockIdxs] = dataBlock.Amplification[usedData]
                    
                    # calculate total number of entries
                    num = num + int(cp.sum(usedData))
                
                except Exception:
                    writeReconstructionLog(
                        f'Data not found for TAS {slE:03d} with emitter number {snE:02d} for motor rotation {mp:02d}.', 3
                    )
                    continue  # continue with next set of data


    # initialized and calculated numbers agree?, otherwise reduce block sizes
    if num != numScans:
        warnings.warn("MissingSenderReceiverInfos: Not all defined data available.")
        writeReconstructionLog('Not all defined sender/receiver available', 3)
        
        usedDataIdxs = cp.arange(0, num)
        
        slBlock, snBlock, rlBlock, rnBlock, mpBlock, gainBlock = removeDataFromArrays(usedDataIdxs, slBlock, snBlock, rlBlock, rnBlock, mpBlock, gainBlock)
        
        AscanBlock = AscanBlock[:, usedDataIdxs]


    return AscanBlock, mpBlock, slBlock, snBlock, snBlock, rlBlock, rnBlock, gainBlock