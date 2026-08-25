import cupy as cp
from types import SimpleNamespace
from loadCalibratedSensorTemperature import loadCalibratedSensorTemperature
from loadTASTemperaturesProcessed import loadTASTemperaturesProcessed
from writeReconstructionLog import writeReconstructionLog
from loadTASTemperatures import loadTASTemperatures
from correctTASTemperatures import correctTASTemperatures
from temperatureToSoundSpeed import temperatureToSoundSpeed

def getTemperatureInfo(path, files, numTAS, rootUniqueID, hardwareVersion):

    #input validation
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if not hasattr(files, '__dict__') and not isinstance(files, dict):
        raise ValueError("'files' must be a struct-like object")

    if not isinstance(numTAS, (int, float, cp.integer, cp.floating)):
        raise ValueError("'numTAS' must be a numeric scalar")

    if not isinstance(rootUniqueID, str):
        raise ValueError("'rootUniqueID' must be a string")

    if not isinstance(hardwareVersion, str):
        raise ValueError("'hardwareVersion' must be a string")
    
    temp = SimpleNamespace()
    # loadCalibratedSensorTemperature returns (temperatures, timeStamps); only
    # the temperature array is used anywhere in this pipeline
    temp.jumoTemp, _ = loadCalibratedSensorTemperature(path, files.tempCaliSensor, rootUniqueID, hardwareVersion)
    jumoAllNaN = False
    
    if not cp.all(cp.isnan(temp.jumoTemp)):
        temp.expectedTemp = cp.mean(temp.jumoTemp[~cp.isnan(temp.jumoTemp)])  # calculate expected mean water temperature
    else:
        jumoAllNaN = True

    tasTempCompUsed = False


    if files.useTASTempComp:

        tasTempCompUsed = True

        try:
            # load temp from files.tempTASComp (standard file, including the preprocessed/corrected temperature data)
            # and extract infos, TemperatureModel4D and TASTemperature
            TASTemp = loadTASTemperaturesProcessed(path, files.tempTASComp)

            # extract TemperatureModel4D
            if hasattr(TASTemp, 'TemperatureModel4D'):
                temp.TemperatureModel4D = TASTemp.TemperatureModel4D
            
            #extract TASTemperatures
            temp.TASTemperature = TASTemp.TASTemperature

        except Exception:
            tasTempCompUsed = False

    
    if not tasTempCompUsed:
        try:
            # load uncorrected temps from file files.tempTAS
            writeReconstructionLog(
                f'{files.tempTASComp} not used. Trying to use TAS temperatures from {files.tempTAS}', 3
            )
            TASTemp = loadTASTemperatures(path, files.tempTAS)
            writeReconstructionLog(f'Found TAS temperatures from {files.tempTAS}', 2)

            # correct temperatures, if requested,
            # otherwise take loaded values
            if files.correctTASTemp:
                temp1 = cp.squeeze(TASTemp.TASTemperature[1, :, :])
                if temp1.ndim == 1 or temp1.shape[0] == 1:
                    temp1 = temp1.T

                temp2 = cp.squeeze(TASTemp.TASTemperature[0, :, :]) 
                if temp2.ndim == 1 or temp2.shape[0] == 1:
                    temp2 = temp2.T

                temp.TASTemperature[1, :, :] = temp1 

                if not jumoAllNaN:
                    writeReconstructionLog('Performing temperature correction based on calibrated temperature sensors', 2)
                    temp.TASTemperature[0, :, :] = correctTASTemperatures(temp2, temp.jumoTemp)
                else:
                    temp.TASTemperature[0, :, :] = temp2
            else:
                temp.TASTemperature = TASTemp.TASTemperature

        except Exception as ME:
            # if TAS temperatures not there
            # take temperatures from calibrated sensors
            writeReconstructionLog(
                f'{type(ME).__name__}: {ME} TAS temperatures approximated from calibrated reference sensors.', 3
            )
            if not jumoAllNaN:
                temp.TASTemperature = cp.zeros((2, numTAS, temp.jumoTemp.shape[1]))
                temp.TASTemperature[0, :, :] = cp.tile(cp.mean(temp.jumoTemp, axis=0), (temp.TASTemperature.shape[1], 1))
                temp.TASTemperature[1, :, :] = cp.tile(
                    cp.arange(1, temp.TASTemperature.shape[1] + 1).reshape(-1, 1),
                    (1, temp.TASTemperature.shape[2])
                )
            else:
                raise ValueError(
                    "loadTASTemperatures:noTemperatureAvailable - "
                    "No TAS and no Jumo temperature available! Cannot reconstruct!"
                )


    if jumoAllNaN:
        allTemps = temp.TASTemperature[0, :, :] 
        temp.expectedTemp = cp.mean(allTemps[~cp.isnan(allTemps)])
        temp.jumoTemp = temp.expectedTemp

    # temp to SOS precalc
    temp.expectedSOSWater = temperatureToSoundSpeed(temp.expectedTemp, 'marczak')


    return temp