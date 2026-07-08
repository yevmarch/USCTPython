import cupy as cp

def rotateAndTranslate(M, pt):
    # validate inputs
    if not isinstance(M, cp.ndarray) or M.ndim != 2 or M.shape[0] != M.shape[1] or not cp.all(cp.isfinite(M)):
        raise ValueError("'M' must be a finite square numeric matrix")

    if not isinstance(pt, cp.ndarray) or pt.ndim != 2 or 1 not in pt.shape or not cp.all(cp.isfinite(pt)):
        raise ValueError("'pt' must be a finite numeric vector")

    transposed = False
    if pt.shape[1] < pt.shape[0]:
        pt = pt.T
        transposed = True

    out = M @ cp.vstack([pt.T, cp.ones((1, pt.shape[0]))])

    # back to normal coordinates
    out = out[:-1, :]

    if transposed:
        out = out.T

    return out