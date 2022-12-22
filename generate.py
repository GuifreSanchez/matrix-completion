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
def genRandomOmega(n, k, OS = 3):
    sizeOmega = int(np.floor(OS * k * (2 * n - k)))
    if (sizeOmega > n*n):
        print("Oversampling parameter too large (OS, sizeOmega, n*n): ", (OS, sizeOmega, n*n))
        return False

    Omega = np.unique(np.random.randint(n, size=(sizeOmega, 2)), axis=0)
    # Since some entries may be repeated in the first randint generation, we keep adding pairs until the
    # length of Omega is as desired
    while len(Omega) < sizeOmega:
        Omega =  np.unique(np.concatenate((Omega, np.random.randint(n, size=(sizeOmega-len(Omega), 2)))), axis=0)
    return Omega

# Function homotopy strategy
def f(x, y, a = 1):
    return 1. / (a + (x - y) * (x - y ))

def square_matrix_from_function(size, function, I0 = 0, I1 = 1):
    M = np.zeros((size,size))
    for i in range(size):
        x = I0 + (I1 - I0) * i / (size - 1)
        for j in range(size):
            y = I0 + (I1 - I0) * j / (size - 1)
            M[i,j] = f(x,y)
    return M

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
def genInitialGuessAsA(n,k):
    X = genRandomMatrixFixedRank(n,k)
    return getCompactSVD(X)

gaussian_factor = 10.
def genInitialGuessSVD(n, k):
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

    return cl.svdForm(U, -np.sort(-np.random.normal(n,n / gaussian_factor, k)), V)

def relativeResidual(X_Omega, A_Omega):
    n = A_Omega.shape[0]
    if (ops.normCSRMatrix(A_Omega) == 0):
        print("Unable to compute relative residual, norm of A Omega is 0")
        return False
    return ops.normCSRMatrix(ops.substractCSR(X_Omega, A_Omega)) / ops.normCSRMatrix(A_Omega)

def LRGeomCG(n, k, Omega, A, X, OS = 3, max_iterations = 100, tol = 1e-12, prints = False, homotopy = False, tol_hom = 1e-3, Lambda = None): 
    t0 = time()   
    t1 = t0
    X_current = X
    A_Omega = ops.getProjection(A, Omega)
    if (prints): print("OS considered: ", OS)
    relChange = 1.0
    relRes = 1.0
    relResHom = 1.0

    iter_results = []
    # Set iter data for i - 1 for the first iteration
    X_prev = X_current
    dir_prev = cl.TangentVector(X_current, np.zeros((k,k)), np.zeros((n,k)), np.zeros((n,k)))
    
    # Set data used to compute CG step in first iteration
    X_Omega_current = ops.projectionFromSVD(X_current,Omega)
    R_sparse_current = ops.substractCSR(X_Omega_current, A_Omega)
    error_current = 0.5 * ops.norm2CSRMatrix(R_sparse_current)
    error_prev = error_current

    if (homotopy):
        A_Lambda = ops.getProjection(A, Lambda)

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
        if (not homotopy):
            relRes = relativeResidual(X_Omega_current, A_Omega)
            iter_results.append(relRes)
            if (prints): print("Relative residual, (rel_res / tol): ", relRes, relRes / tol)
            if relRes <= tol:
                if (prints): print("Relative residual / tol <= 1: ", relRes / tol)
                t1 = time()
                break
        else:
            if (prints): print(relChange, error_current, error_prev, relResHom)
            if (i != 0):
                relChange = abs(1. - sqrt(error_current / error_prev))
                # print("Relative change (iteration, value)", i, relChange)
                if (relChange <= tol_hom):
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
        error_prev = error_current

        X_current = X_retraction
        X_Omega_current = X_retraction_Omega
        R_sparse_current = R_sparse_retraction
        error_current = error_retraction
    
    t1 = time()
    time_elapsed = t1 - t0
    total_iterations = i
    if (homotopy):
        X_Lambda_current = ops.projectionFromSVD(X_current, Lambda)
        relResHom = relativeResidual(X_Lambda_current, A_Lambda)
    return np.array([time_elapsed, total_iterations, relRes, relChange, relResHom, np.array(iter_results), X_current], dtype=object)

# X_prev_opt expected to be passed in svdForm
def genInitGuessHomotopy(n, k, X_prev_opt):
    if (k == 1):
        return genInitialGuessSVD(n,k)
    else: 
        U_prev, Sigma_prev, V_prev = X_prev_opt.U, X_prev_opt.Sigma, X_prev_opt.V

        u = np.random.standard_normal(n)
        v = np.random.standard_normal(n)

        for j in range(0, k - 1):
            uj = U_prev[:,j]
            vj = V_prev[:,j]
            u = u - np.dot(u, uj) * uj
            v = v - np.dot(v, vj) * vj
        
        u = u / sqrt(np.dot(u,u))
        v = v / sqrt(np.dot(v,v))
        # print(U_prev.shape)
        # print(np.array([u]).T.shape)
        # print(Sigma_prev.shape)
        U_new = np.concatenate((U_prev, np.array([u]).T), axis = 1)
        V_new = np.concatenate((V_prev, np.array([v]).T), axis = 1)
        Sigma_new = np.concatenate((Sigma_prev, np.array([Sigma_prev[k - 2]])))
        # print(Sigma_prev)
        # print(Sigma_new)

        # print(u)
        # print(v)
        return cl.svdForm(U_new, Sigma_new, V_new)




INDEX_RESULTS_TIME_ELAPSED = 0
INDEX_RESULTS_ITERATIONS = 1
INDEX_RESULTS_REL_RES = 2
INDEX_RESULTS_REL_CHANGE = 3
INDEX_RESULTS_REL_LAMBDA = 4
INDEX_RESULTS_FULL_ITERS = 5
INDEX_RESULTS_X_OPT = 6

def LRGeomCGGlobal(n, k_max, Omega, Lambda, A_hom, OS = 8, max_iterations = 500, tol = 1e-3, prints = False, homotopy= True):
    X1 = genInitGuessHomotopy(n,1,None)
    Xk = X1
    k_OS = 20
    results = []
    for k in range(1,k_max + 1):
        print("Rank k: ", k)
        if (homotopy):
            if (k != 1):
                Xk = genInitGuessHomotopy(n,k,Xk)
        else:
            Xk = genInitialGuessSVD(n, k)

        #print(Xk.Sigma)
        # Omega = genRandomOmega(n,k_OS,OS)

        results.append(LRGeomCG(n, k, Omega, A_hom, Xk, max_iterations = max_iterations, homotopy = True, OS = 8, Lambda = Lambda, prints = prints))
        print("time_elapsed, iterations, relRes, relChange, relResHom")
        if (homotopy): Xk = results[k - 1][INDEX_RESULTS_X_OPT]
        print(results[k - 1])
    
    return np.array(results)

def LRGeomCGNSamples(n, k, OS = 3, samples = 10, max_iterations = 200, tol = 1e-12, prints = False):
    v_Omega = []
    v_X_init = []
    v_A = []
    v_time_elapsed = []
    v_iterations = []
    results = []
    iter_results = []
    for i in range(samples):
        v_Omega.append(genRandomOmega(n,k, OS = OS))
        v_X_init.append(genInitialGuessSVD(n,k))
        v_A.append(genRandomMatrixFixedRank(n,k))
        results_current = LRGeomCG(n, k, v_Omega[i], v_A[i], v_X_init[i], max_iterations = max_iterations, OS = OS)
        iter_results.append(results_current[INDEX_RESULTS_FULL_ITERS])
        results.append(results_current[:INDEX_RESULTS_REL_LAMBDA])
        # v_time_elapsed.append(time_elapsed)
        # v_iterations.append(iterations)
        if (prints): print("(n, k, sample): ", n, k, i)

    return np.array(results, dtype= object), np.array(iter_results, dtype=object)
    
    
        



 





