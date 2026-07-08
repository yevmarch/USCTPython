import cupy as cp
from estimateBatchSize import estimateBatchSize
from bilinearInterpolationGPU import bilinearInterpolationGPU

def reconstructSAFTimage(ascans, senderPosition, receiverPosition, params):

    # grid vectors describing pixel positions in USCT coordinate system
    xVec = cp.arange(params.startPoint[0], params.endPoint[0] + params.resolution, params.resolution)
    yVec = cp.arange(params.startPoint[1], params.endPoint[1] + params.resolution, params.resolution)
    zVec = cp.arange(params.startPoint[2], params.endPoint[2] + params.resolution, params.resolution)

    xGrid, yGrid, zGrid = cp.meshgrid(xVec, yVec, zVec, indexing='ij')

    gridShape = xGrid.shape
    img = cp.zeros(gridShape)

    # coordinate grids for A-Scan data
    numSamples = ascans.shape[0]
    numAScans = ascans.shape[1]
    ascansGPU = cp.asarray(ascans)

    timeVector = cp.arange(0, numSamples / params.sampleRate, 1 / params.sampleRate)
    ascanNumberVector = cp.arange(1, numAScans + 1)

    # estimation of the size of the batch of AScan block to be processed based on available GPU memory  
    batchSize = estimateBatchSize(gridShape)

    # check whether the elements in timeVector and ascanNumberVector all have an interval of 1
    dx = timeVector[1] - timeVector[0]
    dy = ascanNumberVector[1] - ascanNumberVector[0]

    assert cp.allclose(cp.diff(timeVector), dx), "mapCoordinates require uniform spacing"
    assert cp.allclose(cp.diff(ascanNumberVector), dy), "mapCoordinates require uniform spacing"


    xGrid = xGrid[..., None]
    yGrid = yGrid[..., None]
    zGrid = zGrid[..., None]

    # Reconstruction in batches
    for start in range(0, numAScans, batchSize):
        end = min(start + batchSize, numAScans)

        # calculate distance from sender to all voxels
        distS = cp.sqrt((xGrid - senderPosition[0, start:end])**2 + (yGrid - senderPosition[1, start:end])**2 + (zGrid - senderPosition[2, start:end])**2)

        #calculate distance from receiver to all voxels
        distR = cp.sqrt((xGrid - receiverPosition[0, start:end])**2 + (yGrid - receiverPosition[1, start:end])**2 + (zGrid - receiverPosition[2, start:end])**2)

        tof = (distS + distR) / params.soundSpeed

        # interpolate the amplitudes in the A-Scan
        idxArray = cp.broadcast_to(cp.arange(start + 1, end + 1), tof.shape)
        amp = bilinearInterpolationGPU(dx, dy, tof, idxArray, timeVector, ascanNumberVector, ascansGPU)

        # add interpolated amplitudes to the image
        img += cp.sum(amp, axis=-1)

        # free memory between batches
        del distS, distR, tof, idxArray, amp
        cp.get_default_memory_pool().free_all_blocks()
    
    return img