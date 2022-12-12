import numpy as np
import classes as cl
from scipy.sparse import csr_matrix
import math

def substractCSR(A,B):
    n = A.shape[0]
    # Assumed to be n x n square matrices
    return csr_matrix((A.data - B.data, A.indices, A.indptr), shape=(n,n))

def normCSRMatrix(A):
    return math.sqrt(np.dot(A.data,A.data.transpose()))

def norm2CSRMatrix(A):
    return np.dot(A.data,A.data.transpose())


# Returns CSR matrix that results from projection onto the observation index set Omega
def getProjection(A, Omega):
    # Get dimension of matrix A
    n = A.shape[0]
    # Get array with ordered A[Omega] entries
    flattenedAOmega = np.ravel(A[Omega[:,0],Omega[:,1]])
    # Stored in CSR format
    AOmegaCSR = csr_matrix((flattenedAOmega, (Omega[:,0],Omega[:,1])), shape=A.shape)
    return AOmegaCSR

# Returns CSR matrix
def projectionFromSVD(svdMatrix, Omega):
    return getProjection(svdMatrix.toArray(),Omega)

def innerProduct(xi, eta):
    # Check tangent vectors belong to same tangent space
    cl.checkSameTangentSpace(xi,eta)

    # # Vectorize tangent vectors
    # xiFlatten = xi.toArray().reshape(-1)
    # etaFlatten = eta.toArray().reshape(-1)

    # It can also be computed as M*M' + Up*Up' + Vp*Vp'
    result = np.inner(xi.M.flatten(),eta.M.flatten()) + np.inner(xi.Up.flatten(),eta.Up.flatten()) + np.inner(xi.Vp.flatten(),eta.Vp.flatten())

    return result