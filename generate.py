import numpy as np
import classes as cl
import subroutines as algs
from subroutines import computeConjugateDirection, computeRiemannianGradient, computeRetraction, computeInitialGuessLineSearch, ArmijoBackTracking
import operations as ops
from math import sqrt
from time import time

# Returns random n x n matrix of rank k in np array form
def genRandomMatrixFixedRank(n, k):

    A_L = np.matrix(np.random.standard_normal(size=(n, k)))
    A_R = np.matrix(np.random.standard_normal(size=(k, n)))

    return A_L * A_R

# Returns random n x n integer matrix of rank k in np array form
def genRandomIntMatrixFixedRank(n, k):

    A_L = np.matrix(np.random.randint(n,size=(n, k)))
    A_R = np.matrix(np.random.randint(n,size=(k, n)))

    return A_L * A_R

# Returns np array
# Generates random observation index set Omega, given oversampling (OS) parameter 
def genRandomOmega(n, k, OS =3):
    sizeOmega = OS * k * (2 * n - k)
    if (sizeOmega > n*n):
        print("Oversampling parameter too large (OS, sizeOmega, n*n): ", (OS, sizeOmega, n*n))
        return False

    Omega = np.unique(np.random.randint(n, size=(sizeOmega, 2)), axis=0)
    # Since some entries may be repeated in the first randint generation, we keep adding pairs until the
    # length of Omega is as desired
    while len(Omega) < sizeOmega:
        Omega =  np.unique(np.concatenate((Omega, np.random.randint(n, size=(sizeOmega-len(Omega), 2)))), axis=0)
    return Omega

# Returns SVD object containing the compact SVD decomposition of X
def getCompactSVD(X):
    U, Sigma, VT = np.linalg.svd(X)

    k = np.linalg.matrix_rank(X)
    n = X.shape[0]

    Uk = np.zeros((n,k))
    Vk = np.zeros((n,k))
    Sigmak = np.zeros(k)

    for i in range(n):
        for j in range(k):
            Uk[i,j] = U[i,j]
            Vk[i,j] = VT.T[i,j]
    for j in range(k):
        Sigmak[j] = Sigma[j]

    SVD = cl.svdForm(Uk, Sigmak, Vk)
    return SVD

# Returns svdForm object
def genInitialGuess(n,k):
    X = genRandomMatrixFixedRank(n,k)
    return getCompactSVD(X)

def genInitialGuess2(n, k):
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

def relativeResidual(X_Omega, A_Omega):
    n = A_Omega.shape[0]
    if (ops.normCSRMatrix(A_Omega) == 0):
        print("Unable to compute relative residual, norm of A Omega is 0")
        return False
    return ops.normCSRMatrix(ops.substractCSR(X_Omega, A_Omega)) / ops.normCSRMatrix(A_Omega)

def LRGeomCG(n, k, Omega, A, X, max_iterations = 100, tol = 1e-12, prints = False): 
    t0 = time()   
    t1 = t0
    X_current = X
    A_Omega = ops.getProjection(A, Omega)

    # Set iter data for i - 1 for the first iteration
    X_prev = X_current
    dir_prev = cl.TangentVector(X_current, np.zeros((k,k)), np.zeros((n,k)), np.zeros((n,k)))
    
    # Set data used to compute CG step in first iteration
    X_Omega_current = ops.projectionFromSVD(X_current,Omega)
    R_sparse_current = ops.substractCSR(X_Omega_current, A_Omega)
    error_current = 0.5 * ops.norm2CSRMatrix(R_sparse_current)
    A_Omega_norm = ops.normCSRMatrix(A_Omega)

    for i in range(0,max_iterations):
        if (prints):
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
        if (prints): print("Relative residual, (rel_res / tol): ", relRes, relRes / tol)
        if relRes <= tol:
            if (prints): print("Relative residual / tol <= 1: ", relRes / tol)
            t1 = time()
            break

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
        if (prints): print("CG direction norm: ", dir_current.getNorm())

        # Compute initial guess line search
        t_ast = computeInitialGuessLineSearch(X_current,Omega, dir_current, R_sparse_current)
        if (prints): print ("Initial guess line search: ", t_ast)
        

        # Compute new iterate using Armijo backtracking
        armijo_results = ArmijoBackTracking(X_current, error_current, A_Omega, Omega, t_ast, dir_current, grad_current)
        X_retraction = armijo_results[algs.INDEX_ARMIJO_X_RETR]
        X_retraction_Omega = armijo_results[algs.INDEX_ARMIJO_X_RETR_OMEGA]
        R_sparse_retraction = armijo_results[algs.INDEX_ARMIJO_R_SPARSE_RETR]
        error_retraction = armijo_results[algs.INDEX_ARMIJO_ERROR_RETR]

        X_prev = X_current
        grad_prev = grad_current
        dir_prev = dir_current

        X_current = X_retraction
        X_Omega_current = X_retraction_Omega
        R_sparse_current = R_sparse_retraction
        error_current = error_retraction
    

    time_elapsed = t1 - t0
    total_iterations = i
    return time_elapsed, total_iterations

def LRGeomCGNSamples(n, k, samples = 10, max_iterations = 100, tol = 1e-12, prints = False):
    v_Omega = []
    v_X_init = []
    v_A = []
    v_time_elapsed = []
    v_iterations = []
    for i in range(samples):
        v_Omega.append(genRandomOmega(n,k))
        v_X_init.append(genInitialGuess(n,k))
        v_A.append(genRandomMatrixFixedRank(n,k))
        time_elapsed, iterations = LRGeomCG(n, k, v_Omega[i], v_A[i], v_X_init[i])
        v_time_elapsed.append(time_elapsed)
        v_iterations.append(iterations)
        if (prints): print("Sample ", i, " (time, iters): ", time_elapsed, iterations)

    return np.array(v_time_elapsed), np.array(v_iterations)
    
    
        



 





