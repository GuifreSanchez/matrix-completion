import numpy as np
import classes as cl
import operations as ops
import math

MIN_ALPHA = 0.1
INDEX_ARMIJO_ITER = 0
INDEX_ARMIJO_VALUE = 1
INDEX_ARMIJO_X_RETR = 2
INDEX_ARMIJO_X_RETR_OMEGA = 3
INDEX_ARMIJO_R_SPARSE_RETR = 4
INDEX_ARMIJO_ERROR_RETR = 5

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
    beta =  ops.innerProduct(delta, xi_current) / ops.innerProduct(xi_prev, xi_prev)
    beta = max(0.0, beta)
    new_dir = beta * dir_temp - xi_current  # MAKE SURE THIS OPERATION MAKES SENSE

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
    
    N_data = ops.getProjection(product, Omega).data
    R_sparse_data = R_sparse.data
     # CAREFUL HERE, WE HAVE TO USE A_OMEGA - X_OMEGA!
    t_ast = -1 * np.dot(N_data.transpose(),R_sparse_data) / np.dot(N_data,N_data.transpose())
    return t_ast

def ArmijoBackTracking(X, error_current, A_Omega, Omega, max_t, dir, grad, c = 0.0001, tau = 0.5, max_iters = 50, armijo_tol = 1e-11):

    n = A_Omega.shape[0]
    tau_m = 1.0
    max_dir = max_t * dir
    armijo_right_factor = c * max_t * ops.innerProduct(grad, dir)
    for m in range(max_iters):
        print("Armijo iterates: ", m)
        X_retraction =computeRetraction(X, tau_m * max_dir)
        X_retraction_Omega = ops.projectionFromSVD(X_retraction, Omega)
        R_sparse_retraction = ops.substractCSR(X_retraction_Omega, A_Omega)
        error_retraction = 0.5 * ops.norm2CSRMatrix(R_sparse_retraction)
        armijo_value = error_current - error_retraction + tau_m * armijo_right_factor
        if armijo_value >= 0:
            return [m, armijo_value, X_retraction, X_retraction_Omega, R_sparse_retraction, error_retraction]
        tau_m *= tau
    
    print("Error: Armijo condition not satisfied")
    return False

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