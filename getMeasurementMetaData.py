from loadMeasurementInfos import loadMeasurementInfos
from detectAScanLength import detectAScanLength
from getTransformationMatrix import getTransformationMatrix
from getTemperatureInfo import getTemperatureInfo
from getCEInfo import getCEInfo

def getMeasurementMetaData(pathToMeasurement, files, motorPos, numTAS, paramsPreprocessing, rootUniqueID):

    #loading measurement metadata
    info = loadMeasurementInfos(pathToMeasurement, files.info, [], rootUniqueID)

    #trying to find out the data size from measured A-Scans and setting according parameters
    ascanLength = detectAScanLength(pathToMeasurement, info, paramsPreprocessing)
    if len(ascanLength) == 0:
        ascanLength = paramsPreprocessing.expectedAScanLength
    info.expectedAScanLength = ascanLength

    #get transformation matrices of USCT aperture positions
    rotationMatrix, motorPos = getTransformationMatrix(pathToMeasurement, files, motorPos)

    #get temperature information
    temp = getTemperatureInfo(pathToMeasurement, files.temp, numTAS, rootUniqueID, info.Hardware)

    #get CE information
    ce = getCEInfo(pathToMeasurement, files.ce, info, paramsPreprocessing)

    return info, temp, ce, rotationMatrix, motorPos