import cupy as cp

def estimateBatchSize(gridShape, dtype=cp.float64, safety_factor=0.3):
    freeBytes, totalBytes = cp.cuda.device().mem_info
    usableBytes = freeBytes * safety_factor

    voxelsPerScan = 1
    for i in gridShape:
        voxelsPerScan *= i

    bytesPerScan = voxelsPerScan * cp.dtype(dtype).itemsize
    bytesPerScanTotal = bytesPerScan * 5

    blockSize = max(1, int(usableBytes // bytesPerScanTotal))

    return blockSize