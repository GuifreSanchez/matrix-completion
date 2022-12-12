import numpy as np
import classes as cl
import operations as ops
import math
from scipy.sparse import csr_matrix
from math import sqrt


MIN_ALPHA = 0.1

# Returns matrix in np array form
def genRandomMatrixFixedRank(n, k):

    A_L = np.matrix(np.random.standard_normal(size=(n, k)))
    A_R = np.matrix(np.random.standard_normal(size=(k, n)))

    return A_L * A_R

def genRandomIntMatrixFixedRank(n, k):

    A_L = np.matrix(np.random.randint(n,size=(n, k)))
    A_R = np.matrix(np.random.randint(n,size=(k, n)))

    return A_L * A_R

# Returns np array
def genRandomOmega(n, k, OS =3):
    sizeOmega = OS * k * (2 * n - k)
    if (sizeOmega > n*n):
        print("Oversampling parameter too large (OS, sizeOmega, n*n): ", (OS, sizeOmega, n*n))
        return False

    Omega = np.unique(np.random.randint(n, size=(sizeOmega, 2)), axis=0)
    while len(Omega) < sizeOmega:
        Omega =  np.unique(np.concatenate((Omega, np.random.randint(n, size=(sizeOmega-len(Omega), 2)))), axis=0)
    return Omega

def getCompactSVD(X):
    U, Sigma, VT = np.linalg.svd(X)
    rank = np.linalg.matrix_rank(X)
    U = U[:,:rank]
    V = VT.T[:,:rank]
    Sigma = Sigma[:rank]
    SVD = cl.svdForm(U, Sigma, V)
    return SVD


# Returns svdForm object
def genInitialGuess(n,k):
    X = genRandomMatrixFixedRank(n,k)
    SVD = getCompactSVD(X)

    # print("Initial guess shapes U, Sigma, V", SVD.U.shape, SVD.Sigma.shape, SVD.V.shape)
    # print("Rank: ", np.linalg.matrix_rank(X))

    return SVD

# Returns CSR matrix
def projectionMatrix(A, Omega):
    # Get dimension of matrix A
    n = A.shape[0]
    # print(n)
    # Get array with ordered A[Omega] entries
    flattenedAOmega = np.ravel(A[Omega[:,0],Omega[:,1]])

    # print(flattenedAOmega)
    # print(len(flattenedAOmega))
    # print(Omega[:,0])
    # print(len(Omega[:,0]))

    # Stored in CSR format
    AOmegaCSR = csr_matrix((flattenedAOmega, (Omega[:,0],Omega[:,1])), shape=A.shape)
    return AOmegaCSR

# Returns CSR matrix
def projectionFromSVD(svdMatrix, Omega):
    return projectionMatrix(svdMatrix.toArray(),Omega)

# ALGORITHM 2
# Returns tangent vector with gradient
def computeRiemannianGradient(X, RSparse):
    # Multiply sparse x dense matrix efficiently
    Ru = RSparse.transpose().dot(X.U)
    Rv = RSparse.dot(X.V)
    M = X.U.T @ Rv
    Up = Rv - X.U @ M
    Vp = Ru - X.V @ M.T
    return cl.TangentVector(X,M,Up,Vp)

# ALGORITHM 3
# Returns tangent vector with transported vector
def transportVector(X1, X2, nu):
    U1, S1, V1 = X1.U, X1.Sigma, X1.V
    U2, S2, V2 = X2.U, X2.Sigma, X2.V

    M, Up, Vp = nu.M, nu.Up, nu.Vp

    # print("Shapes V1, V2: ", V1.shape, V2.shape)
    Av, Au = V1.T@V2, U1.T@U2
    Bv, Bu = Vp.T@V2, Up.T@U2


    # print("Shapes Au, M, Av: ", Au.shape, M.shape, Av.shape)
    M12, U12, V12 = Au.T@M@Av, U1@(M@Av), V1@(M.T@Au)
    M22, U22, V22 = Bu.T@Av, Up@Av, V1@Bu
    M32, U32, V32 = Au.T@Bv, U1@Bv, Vp@Au

    M2 = M12 + M22 + M32

    Up2 = U12 + U22 + U32
    Up2 = Up2 - U2@(U2.T@Up2)

    Vp2 = V12 + V22 + V32
    Vp2 = Vp2 - V2@(V2.T@Vp2)

    nu2 = cl.TangentVector(X2,M2,Up2,Vp2)
    return nu2

# ALGORITHM 4 COMPUTE CONJUGATE DIRECTION BY PR+
# x_prev, x_current are points in M_k
# xi_prev, dir_prev are vectors in T_(x_prev)M_k
# xi_current is a vector in T_(x_current)M_k
def computeConjugateDirection(x_prev, x_current, xi_prev, xi_current, dir_prev):
    # Transport previous gradient and direction to current tangent space
    xi_temp = transportVector(x_prev, x_current, xi_prev)
    dir_temp = transportVector(x_prev, x_current, dir_prev)

    # Compute conjugate direction
    delta = xi_current - xi_temp
    xi_prev_norm2 = ops.innerProduct(xi_prev, xi_prev)
    beta = max(0,ops.innerProduct(delta, xi_current) / xi_prev_norm2) 
    dir_temp_beta = dir_temp * beta
    new_dir = dir_temp_beta - xi_current  # MAKE SURE THIS OPERATION MAKES SENSE

    # Compute angle between conjugate direction and gradient
    alpha = ops.innerProduct(xi_current, new_dir) / math.sqrt(ops.innerProduct(xi_current, xi_current) * ops.innerProduct(new_dir, new_dir))

    # Reset to gradient if angle is too low: # CHECK IF ABS IS NEEDED
    if abs(alpha) <= MIN_ALPHA: 
         new_dir = xi_current

    return new_dir

# ALGORITHM 5
# Compute initial guess for line search
# X is in svdForm, X_Omega and R_sparse = X_Omega - A_Omega are csr_matrix objects
# eta is a tangent vector object representing a tangent vector in T_X M_k
# Function returns a scalar
def computeInitialGuessLineSearch(X, Omega, eta, R_sparse):
    # # Concatenate matrix one after the other left to right
    # A = genRandomIntMatrixFixedRank(3,2)
    # B = genRandomIntMatrixFixedRank(3,2)
    # C = np.concatenate((A,B), axis=1)
    # print("A: \n", A, "\n B: \n" , B, "\n C: \n", C)

    block_1 = np.concatenate((X.U@eta.M + eta.Up,X.U), axis = 1)
    block_2 = np.concatenate((X.V,eta.Vp), axis = 1)
    product = block_1@block_2.T
    
    N_data = projectionMatrix(product, Omega).data
    R_sparse_data = R_sparse.data
     # CAREFUL HERE, WE HAVE TO USE A_OMEGA - X_OMEGA!
    t_ast = -1 * np.dot(N_data.transpose(),R_sparse_data) / np.dot(N_data,N_data.transpose())
    return t_ast

def costFunction(X, A, Omega):
    X_Omega = projectionFromSVD(X,Omega)
    A_Omega = projectionMatrix(A, Omega)
    n = A.shape[0]
    R_sparse = substractCSR(X_Omega, A_Omega)
    R_frob_norm = np.dot(R_sparse.data, R_sparse.data.transpose())
    return 0.5 * R_frob_norm

def generateInitialGuess2(n, k):
    U = np.zeros((n,k))
    V = np.zeros((n,k))

    for i in range(k):
        u = np.random.standard_normal(n)
        v = np.random.standard_normal(n)
        for j in range(0,i):
            u = u - np.dot(u, U[:,j])*U[:,j]
            v = v - np.dot(v, V[:,j])*V[:,j]
    
        U[:,i] = u/sqrt(np.dot(u,u))
        V[:,i] = v/sqrt(np.dot(v,v))

    return cl.svdForm(U, np.random.standard_normal(k), V)


INDEX_ARMIJO_ITER = 0
INDEX_ARMIJO_VALUE = 1
INDEX_ARMIJO_X_RETR = 2
INDEX_ARMIJO_X_RETR_OMEGA = 3
INDEX_ARMIJO_R_SPARSE_RETR = 4
INDEX_ARMIJO_ERROR_RETR = 5
def ArmijoBackTracking(X, error_current, A_Omega, Omega, max_t, dir, grad, c = 0.0001, tau = 0.5, max_iters = 50, armijo_tol = 1e-11):

    n = A_Omega.shape[0]
    tau_m = 1.0
    max_dir = max_t * dir
    armijo_right_factor = c * max_t * ops.innerProduct(grad, dir)
    for m in range(max_iters):
        print("Armijo iterates: ", m)
        X_retraction =computeRetraction(X, tau_m * max_dir)
        X_retraction_Omega = projectionFromSVD(X_retraction, Omega)
        R_sparse_retraction = substractCSR(X_retraction_Omega, A_Omega)
        error_retraction = 0.5 * norm2CSRMatrix(R_sparse_retraction)
        armijo_value = error_current - error_retraction + tau_m * armijo_right_factor
        if armijo_value >= 0:
            return [m, armijo_value, X_retraction, X_retraction_Omega, R_sparse_retraction, error_retraction]
        tau_m *= tau
    
    print("Error: Armijo condition not satisfied")
    return False

    
    


    # tauj = tau
    # gradDotXi = ops.innerProduct(grad, dir)
    # #print("Cost function values: ", fX, costFunction(computeRetraction(X,dir.scalar(max_t)), A, Omega))
    # for m in range(max_iters):
    #     backTrackingStep = (0.5**m) * max_t 
    #     linePoint = dir.scalar(backTrackingStep)
    #     X_retraction = computeRetraction(X,linePoint)
    #     X_retraction_Omega = projectionFromSVD(X_retraction, Omega)
    #     R_sparse_retraction = substractCSR(X_retraction_Omega, A_Omega)
    #     fRetraction = 0.5 * np.dot(R_sparse_retraction.data, R_sparse_retraction.data.transpose())
    #     ArmijoCondition = fX - fRetraction + c * backTrackingStep * gradDotXi
    #     print("Armijo iter, value: ", m + 1, ArmijoCondition)
    #     if (ArmijoCondition >= 0):
    #         return [m + 1, ArmijoCondition, X_retraction, X_retraction_Omega, R_sparse_retraction]
    #     else:
    #         tauj = tauj * tau

    # print("Error: Armijo condition not met")
    #return [m + 1, 0, X_retraction, X_retraction_Omega, R_sparse_retraction]


# ALGORITHM 6
# Compute retraction by metric projection
def computeRetraction(X, xi):
    # # Trying QR decomposition
    # # It works for non-square matrices
    # A = genRandomIntMatrixFixedRank(4,2)[:,:3]
    # q, r = np.linalg.qr(A)
    # print("q:\n", q)
    # print("r:\n", r)
    # print("A:\n", A)
    # print("A recover:\n", q@r)
    Qu, Ru = np.linalg.qr(xi.Up)
    Qv, Rv = np.linalg.qr(xi.Vp)

    # Assemble S matrix
    #print("Up shape: ", xi.Up.shape)
    k = xi.Up.shape[1]
    block1 = np.concatenate((np.diag(X.Sigma) + xi.M, Rv.T), axis = 1)
    block2 = np.concatenate((Ru, np.zeros((k,k))), axis = 1)
    S = np.concatenate((block1, block2), axis = 0)

    # SVD S
    Us, SigmaS, VsT = np.linalg.svd(S)

    # SVD Retraction
    Uplus = np.concatenate((X.U, Qu), axis = 1) @ Us[:,:k]
    Vplus = np.concatenate((X.V, Qv), axis = 1) @ VsT.transpose()[:,:k]
    Sigmaplus = SigmaS[:k] + 1e-16*np.ones(k)

    retraction = cl.svdForm(Uplus, Sigmaplus, Vplus)

    return retraction

def substractCSR(A,B):
    n = A.shape[0]
    # Assumed to be n x n square matrices
    return csr_matrix((A.data - B.data, A.indices, A.indptr), shape=(n,n))

def normCSRMatrix(A):
    return math.sqrt(np.dot(A.data,A.data.transpose()))

def norm2CSRMatrix(A):
    return np.dot(A.data,A.data.transpose())

def relativeResidual(X_Omega, A_Omega):
    n = A_Omega.shape[0]
    if (normCSRMatrix(A_Omega) == 0):
        print("Unable to compute relative residual, norm of A Omega is 0")
        return False
    return normCSRMatrix(substractCSR(X_Omega, A_Omega)) / normCSRMatrix(A_Omega)

def LRGeomCG(n, k, Omega, A, X1, max_iterations = 100, tol = 1e-12):
    X_current = X1
    A_Omega = projectionMatrix(A, Omega)

    # Set iter data for i - 1 for the first iteration
    X_prev = X_current
    dir_prev = cl.TangentVector(X_current, np.zeros((k,k)), np.zeros((n,k)), np.zeros((n,k)))
    
    # Set data used to compute CG step in first iteration
    X_Omega_current = projectionFromSVD(X_current,Omega)
    R_sparse_current = substractCSR(X_Omega_current, A_Omega)
    error_current = 0.5 * norm2CSRMatrix(R_sparse_current)
    A_Omega_norm = normCSRMatrix(A_Omega)

    for i in range(0,max_iterations):
        print("\n")
        print("ITERATION ", i)
        print("=========================")
        # Compute gradient
        grad_current = computeRiemannianGradient(X_current, R_sparse_current)

        # Check tolerance
        # # Stopping criteria using gradient norm
        # grad_norm_current = grad_current.getNorm()
        # if grad_norm_current <= tol:
        #     print("Grad norm / tolerance is <= 1: ", grad_norm_current / tol)
        #     return True

        # Stopping criteria using relative residuals
        relRes = relativeResidual(X_Omega_current, A_Omega)
        print("Relative residual, (rel_res / tol): ", relRes, relRes / tol)
        if relRes <= tol:
            print("Relative residual / tol <= 1: ", relRes / tol)
            return True

        # # Stopping criteria Mark
        # grad_current_norm = grad_current.getNorm() 
        # print("Gradient norm: ", grad_current_norm)
        # print("A_Omega norm: ", A_Omega_norm)
        # print("Grad norm / A_Omega norm: ", grad_current_norm / A_Omega_norm)
        # if grad_current_norm <= tol * A_Omega_norm:
        #     print("Custom rel. residual / tol <= 1: ", grad_current_norm / A_Omega_norm, grad_current_norm / A_Omega_norm / tol * 100)
        #     return True
        
        # Compute conjugate direction by PR+

        # For the first iteration, we set grad[i - 1] = grad[0]
        if (i == 0):
            grad_prev = grad_current

        dir_current = computeConjugateDirection(X_prev, X_current, grad_prev, grad_current, dir_prev)
        print("CG direction norm: ", dir_current.getNorm())

        # Compute initial guess line search
        t_ast = computeInitialGuessLineSearch(X_current,Omega, dir_current, R_sparse_current)
        print ("Initial guess line search: ", t_ast)
        

        # Compute new iterate using Armijo backtracking
        armijo_results = ArmijoBackTracking(X_current, error_current, A_Omega, Omega, t_ast, dir_current, grad_current)
        X_retraction = armijo_results[INDEX_ARMIJO_X_RETR]
        X_retraction_Omega = armijo_results[INDEX_ARMIJO_X_RETR_OMEGA]
        R_sparse_retraction = armijo_results[INDEX_ARMIJO_R_SPARSE_RETR]
        error_retraction = armijo_results[INDEX_ARMIJO_ERROR_RETR]

        X_prev = X_current
        grad_prev = grad_current
        dir_prev = dir_current

        X_current = X_retraction
        X_Omega_current = X_retraction_Omega
        R_sparse_current = R_sparse_retraction
        error_current = error_retraction
        



 





