

import classes  as cl
import generate as gen
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from math import sqrt
from time import time

# n = 3
# k = 1
# OS = 1

# Omega = gen.genRandomOmega(n,k,OS)
# sizeOmega = len(Omega)

# A = gen.genRandomMatrixFixedRank(n,k)
# rankA = np.linalg.matrix_rank(A)

# B = gen.genRandomIntMatrixFixedRank(n,k)
# BSVD = cl.getSVDForm(B)
# BRecover = BSVD.toArray()

# initGuess = gen.genInitialGuess(n,k)

# print("This matrix has rank", rankA)
# print("Omega length: ", sizeOmega)
# print("Omega: \n", Omega)
# print("Random int matrix: \n", B)
# print("B Omega: \n", B[Omega[:,0],Omega[:,1]].reshape(-1))
# print("SVD Form of int matrix: \n", BSVD)
# print("Recover from SVD Form: \n", np.array(BRecover))
# print("Initial guess: ", initGuess)

n = 1000
k = 40

A = gen.genRandomMatrixFixedRank(n, k)
X0 = gen.genInitialGuess(n,k)
X1 = gen.generateInitialGuess2(n,k)
Omega = gen.genRandomOmega(n, k)
print("Length Omega: ", len(Omega))

gen.LRGeomCG(n,k,Omega, A, X1)

#gen.computeInitialGuessLineSearch(1,2,3,4)
#gen.computeRetraction(1,2)