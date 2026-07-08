from loadFileToStruct import loadFileToStruct
from convertGeometry import convertGeometry

def loadGeometry(file):

    # predefine variables
    loadVariable = 'TASElements'

    # input validation
    if file is None or file == '':
        file = 'geometryFileUSCT3Dv2_4.mat'
    
    if not isinstance(file, str) or not file:
        raise ValueError("'file' must be a non-empty string")
    
    # load requested variable from onfo file
    geom = loadFileToStruct(file, loadVariable)
    geom = getattr(geom, loadVariable)

    # check if all required fields are available (if not, expected to be usct 3 geometry)
    # try to convert to an expected format
    requiredFields = ['emitterNormals', 'receiverNormals', 'emitterPositions', 'receiverPositions']
    if not all(hasattr(geom, f) for f in requiredFields):
        try:
            geom = convertGeometry(geom)
        except Exception:
            raise ValueError(
                "loadGeometry:variablesNotFound - "
                "Unexpected geometry format: Geometry file does not contain required TAS informations."
            )
    
    return geom