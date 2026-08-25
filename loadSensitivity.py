from loadFileToStruct import loadFileToStruct
from interpn_2d import interpn_2d
import cupy as cp

def loadSensitivity(file, usctVersion):

    # predefine a variable
    loadVariable = 'angleCorrection'

    # input validation
    if file is None or file == '':
        file = 'transduce_angle_char.mat'

    if not isinstance(file, str):
        raise ValueError("'file' must be a non-empty string")
    
    # load requested variable from file (sensitivity)
    sens = loadFileToStruct(file, loadVariable)

    # check variable and generate output
    if sens is None or not hasattr(sens, loadVariable):
        raise ValueError(
            "loadSens:variablesNotFound - "
            "Sensitivity file not found or does not contain required variables!")
    
    sens = getattr(sens, loadVariable)

    # angleCorrection is stored as a 2D row vector (e.g. shape (1, 101)), not
    # 1D -- the code right below this needs two dimensions (sens.shape[0/1])
    if not isinstance(sens, cp.ndarray) or sens.ndim != 2:
        raise ValueError("'Sensitivity' must be a numeric vector")
    
    if sens.shape[0] > sens.shape[1]:
        sens = sens.T

    #sensitivity data is given from -90 to 0 degrees

    fullSens1D = cp.concatenate([sens, sens[:, -2::-1]], axis=1)
    szFullSens1D = fullSens1D.shape[1]


    # mirror to get full angle coverage

    # USCT II: extend to 2D by using quadratic symmetry
    if usctVersion.lower() == 'usct3dv2':
        fullSens2D = cp.tile(fullSens1D, (szFullSens1D, 1))
        fullSens2DTransposed = fullSens2D.T
        fullSens2D = cp.sqrt(fullSens2D * fullSens2DTransposed)
        fullSens2D = fullSens2D / cp.max(fullSens2D)

    # USCT III: extend to 2D by rotational symmetry
    elif usctVersion.lower() == 'usct3dv3':
        # fullSens1D is a (1, szFullSens1D) row vector -- len() on it returns
        # 1 (the number of rows), not szFullSens1D, so reuse the already
        # correctly computed length; likewise slice columns (axis=1) below,
        # not rows
        n = szFullSens1D // 2
        r = cp.concatenate([cp.arange(n, -1, -1), cp.arange(1, n + 1)])
        ri = cp.sqrt(r[:, None]**2 + r[None, :]**2)
        x_known = r[:n + 1][::-1]              # reverse to ascending order
        y_known = fullSens1D[:, :n + 1].ravel()[::-1]  # reverse to match
        
        fullSens2D = cp.interp(ri.ravel(), x_known, y_known, left=cp.nan, right=cp.nan).reshape(ri.shape)
        fullSens2D[cp.isnan(fullSens2D)] = 0

    #finally interpolate to grid in steps of 1° from -90° to +90° -> 181 angle steps

    x0, y0 = cp.meshgrid(
        cp.linspace(-90, 90, szFullSens1D),
        cp.linspace(-90, 90, szFullSens1D),
        indexing='ij'
    )
    x1, y1 = cp.meshgrid(
        cp.linspace(-90, 90, 181),
        cp.linspace(-90, 90, 181),
        indexing='ij'
    )

    sens = interpn_2d(x0, y0, fullSens2D, x1, y1)

    return sens