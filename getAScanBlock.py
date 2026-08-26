import cupy as cp
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from loadAscanData import loadAscanData
from checkForCommonListEntries import checkForCommonListEntries
from writeReconstructionLog import writeReconstructionLog
from removeDataFromArrays import removeDataFromArrays
import warnings

def getAScanBlock(path, mp, sl, sn, rl ,rn, measInfo, rootUniqueID, maxWorkers=16):
    
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

    AscanBlock = cp.zeros((int(measInfo.NumberSamples), numScans), dtype=dataType)
    slBlock = cp.zeros((1, numScans), dtype=cp.int16)
    snBlock = cp.zeros((1, numScans), dtype=cp.int16)
    rlBlock = cp.zeros((1, numScans), dtype=cp.int16)
    rnBlock = cp.zeros((1, numScans), dtype=cp.int16)
    mpBlock = cp.zeros((1, numScans), dtype=cp.int16)
    gainBlock = cp.ones((1, numScans))

    # cupy iteration yields 0-d arrays rather than native scalars (unlike
    # numpy); cast to plain int up front so loadAscanData's isinstance(...,
    # int) scalar checks don't reject them. Order matches the original
    # nested loop (mp outer, sl middle, sn inner) so block indexing below is
    # unaffected.
    combos = [
        (int(mpE), int(slE), int(snE))
        for mpE in mp.flatten()
        for slE in sl.flatten()
        for snE in sn.flatten()
    ]

    # each combo is an independent file (different TAS/emitter/rotation), and
    # loading is I/O-bound (reading .mat files, often over a network-mounted
    # drive) -- so threads help despite the GIL. Load them all concurrently
    # first; the block-writing aggregation below stays sequential, since it
    # depends on a running byte offset (num) that must advance in a fixed
    # order, and cupy arrays aren't safe for concurrent writes anyway.
    def _load(combo):
        mpE, slE, snE = combo
        try:
            return loadAscanData(path, slE, snE, mpE, measInfo.Hardware, rootUniqueID)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=maxWorkers) as executor:
        dataBlocks = list(executor.map(_load, combos))

    for (mpE, slE, snE), dataBlock in zip(combos, dataBlocks):
        if dataBlock is None:
            writeReconstructionLog(
                f'Data not found for TAS {slE:03d} with emitter number {snE:02d} for motor rotation {mpE:02d}.', 3
            )
            continue

        try:
            # check for required receiver data
            # (flatten: checkForCommonListEntries returns a (N,1)
            # column-vector mask, and AscanBlock[:, usedData] below
            # mixes a leading ":" slice with a 2D boolean mask, which
            # numpy/cupy reject even though usedData alone works fine)
            usedData = checkForCommonListEntries(dataBlock.TASIndices, rl, dataBlock.receiverIndices, rn).ravel()

            # calculations for block writings
            sizeDataBlock = int(cp.sum(usedData))
            blockIdxs = slice(num, sizeDataBlock + num)

            # write required data to blocks
            mpBlock[0, blockIdxs] = mpE
            slBlock[0, blockIdxs] = slE
            snBlock[0, blockIdxs] = snE
            # TASIndices/receiverIndices/Amplification are (N,1) column
            # vectors; masking with the now-1D usedData preserves that
            # trailing size-1 axis, so ravel to match the 1D block rows
            rlBlock[0, blockIdxs] = dataBlock.TASIndices[usedData].ravel()
            rnBlock[0, blockIdxs] = dataBlock.receiverIndices[usedData].ravel()
            AscanBlock[:, blockIdxs] = dataBlock.AScans[:, usedData]

            gainBlock[0, blockIdxs] = dataBlock.Amplification[usedData].ravel()

            # calculate total number of entries
            num = num + int(cp.sum(usedData))

        except Exception:
            writeReconstructionLog(
                f'Data not found for TAS {slE:03d} with emitter number {snE:02d} for motor rotation {mpE:02d}.', 3
            )
            continue  # continue with next set of data


    # initialized and calculated numbers agree?, otherwise reduce block sizes
    if num != numScans:
        warnings.warn("MissingSenderReceiverInfos: Not all defined data available.")
        writeReconstructionLog('Not all defined sender/receiver available', 3)
        
        usedDataIdxs = cp.arange(0, num)
        
        slBlock, snBlock, rlBlock, rnBlock, mpBlock, gainBlock = removeDataFromArrays(usedDataIdxs, slBlock, snBlock, rlBlock, rnBlock, mpBlock, gainBlock)
        
        AscanBlock = AscanBlock[:, usedDataIdxs]


    return AscanBlock, mpBlock, slBlock, snBlock, rlBlock, rnBlock, gainBlock
