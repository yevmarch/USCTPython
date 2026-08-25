import cupy as cp
from types import SimpleNamespace
from pathlib import Path
from readConfigFile import readConfigFile
from getUniqueID import getUniqueID
from writeReconstructionLog import writeReconstructionLog
from getMeasurementMetaData import getMeasurementMetaData
from getGeometryInfo import getGeometryInfo
from createMatchedFilter import createMatchedFilter
from estimateOffset import estimateOffset

def getMetaData(path):
    params = SimpleNamespace()
    params.reConfig = Path('configurationFiles') / 'configReconstruction.json'
    params.reflectConfig = Path('configurationFiles') / 'configReflectionReconstruction.json'

    recoParams = readConfigFile(params.reConfig)
    reflectParams = readConfigFile(params.reflectConfig)

    motorPosTotal = reflectParams.dataSelection.motorPos
    slList = cp.array(reflectParams.dataSelection.senderTasList, dtype=cp.uint16)
    snList = cp.array(reflectParams.dataSelection.senderElementList, dtype=cp.uint16)
    rlList = cp.array(reflectParams.dataSelection.receiverTasList, dtype=cp.uint16)
    rnList = cp.array(reflectParams.dataSelection.receiverElementList, dtype=cp.uint16)

    params.rootMeasUniqueID = getUniqueID(path, 'info.mat')

    maxNumTAS = int(cp.maximum(cp.max(rlList), cp.max(slList)))

    expInfo, temp, ce, transformationMatrices, motorPosAvailable = getMeasurementMetaData(path, recoParams.measurementInfo, motorPosTotal, maxNumTAS, reflectParams.dataPreparation, params.rootMeasUniqueID)

    transformationMatricesRef = cp.array([[]])
    motorPosAvailableRef = cp.array([[]])

    geom = getGeometryInfo(recoParams.settingInfo, motorPosAvailable, motorPosAvailableRef, rlList, rnList, slList, snList, transformationMatrices, transformationMatricesRef, expInfo.Hardware)

    if not reflectParams.dataPreparation.aScanReconstructionFrequency and (expInfo.SampleRate != reflectParams.dataPreparation.aScanReconstructionFrequency):
        writeReconstructionLog('Update expectedAScanSampFreq & AScan length to requested up-sample value', 2)
        reflectParams.dataPreparation.expectedAScanLength = cp.ceil(expInfo.NumberSamples*(reflectParams.dataPreparation.aScanReconstructionFrequency / expInfo.SampleRate))
        reflectParams.dataPreparation.aScanReconstructionFrequency = reflectParams.dataPreparation.aScanReconstructionFrequency
    else:
        reflectParams.dataPreparation.aScanReconstructionFrequency = expInfo.SampleRate

    #matched filtering
    preComputes = SimpleNamespace()
    if hasattr(ce, 'CE'):
        preComputes.matchedFilter = createMatchedFilter(ce.CE, ce.measuredCEused, reflectParams.dataSelection, recoParams.measurementInfo.ce.removeOutliersFromCEMeasured, expInfo.Hardware)
    else:
        preComputes.matchedFilter = createMatchedFilter(ce.CERef, ce.measuredCEused, reflectParams.dataSelection, recoParams.measurementInfo.ce.removeOutliersFromCEMeasured, expInfo.Hardware)
        
    preComputes.TimeInterval = 1 / (reflectParams.dataPreparation.aScanReconstructionFrequency)
    preComputes.measuredCEused = ce.measuredCEused
    if(ce.measuredCEused):
        preComputes.measuredCE_TASIndices = ce.TASIndices
        preComputes.measuredCE_receiverIndices = ce.receiverIndices
    preComputes.offset = estimateOffset(reflectParams, expInfo, ce, preComputes.matchedFilter)

    return params, recoParams, reflectParams, expInfo, temp, ce, transformationMatrices, motorPosAvailable, geom, preComputes
