import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

def bilinearInterpolationGPU(dx, dy, xq, yq, xAxis, yAxis, aScansGPU):
    #dx = xAxis[1] - xAxis[0]
    #dy = yAxis[1] - yAxis[0]

    # check uniform spacing

    #assert cp.allclose(cp.diff(xAxis), dx), "mapCoordinates require uniform spacing"
    #assert cp.allclose(cp.diff(yAxis), dy), "mapCoordinates require uniform spacing"

    x_idx = (xq - xAxis[0]) / dx
    y_idx = (yq - yAxis[0]) / dy

    coordinates = cp.stack([x_idx.ravel(), y_idx.ravel()])

    result = map_coordinates(aScansGPU, coordinates, order=1, mode="constant", cval=cp.nan)

    return result.reshape(xq.shape)