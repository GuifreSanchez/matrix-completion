import numpy as np
import classes as cl
import generate as gen


def innerProduct(xi, eta):
    # Check tangent vectors belong to same tangent space
    cl.checkSameTangentSpace(xi,eta)

    # # Vectorize tangent vectors
    # xiFlatten = xi.toArray().reshape(-1)
    # etaFlatten = eta.toArray().reshape(-1)

    # It can also be computed as M*M' + Up*Up' + Vp*Vp'
    result = np.inner(xi.M.flatten(),eta.M.flatten()) + np.inner(xi.Up.flatten(),eta.Up.flatten()) + np.inner(xi.Vp.flatten(),eta.Vp.flatten())

    return result