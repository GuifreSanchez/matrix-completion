import numpy as np
import operations as ops
import math

# SVD Form class
class svdForm:
    def __init__(self, U, Sigma, V):
        # m x k matrix
        self.U = U
        # array of k elements
        self.Sigma = Sigma
        # n x k matrix
        self.V = V

    def __str__(self):
        return "U -> \n" +  str(self.U) +  "\n Sigma -> \n"+  str(self.Sigma) + "\n V -> \n" + str(self.V)

    def __eq__(self, other):
        return (self.U == other.U).all() and (self.Sigma == other.Sigma).all() and (self.V == other.V).all()

    def toArray(self):
        return self.U@np.diag(self.Sigma)@self.V.T

# Get SVD Form 
def getSVDForm(A):
    U, Sigma, VT = np.linalg.svd(A)
    return svdForm(U, Sigma, VT.T)

# Tangent vector class
class TangentVector:
    def __init__(self, X, M, Up, Vp): 
        # X passed as svdForm object  
        self.X = X
        self.M = M
        self.Up = Up
        self.Vp = Vp

    def scalar(self, c):
        c = np.array(c).item()
        return TangentVector(self.X, self.M * c, self.Up * c, self.Vp * c)

    def __sub__(self, other):           ##### TODO! check of return
        if (self.X != other.X):
            print("Tangent vectors from different tangent spaces. Subtraction not allowed.")
            return False
            
        return TangentVector(self.X, self.M - other.M, self.Up - other.Up, self.Vp - other.Vp)
    
    def __add__(self, other):
        if (self.X != other.X):
                print("Tangent vectors from different tangent spaces. Subtraction not allowed.")
                return False
        return TangentVector(self.X, self.M + other.M, self.Up + other.Up, self.Vp + other.Vp)

    def toArray(self):
        return self.X.U @self.M @ self.X.V.T + self.Up @ self.X.V.T + self.X.U @ self.Vp.T

    def getNorm(self):
        return math.sqrt(ops.innerProduct(self,self))

    def __mul__(self, other):
        return TangentVector(self.X, other*self.M, other*self.Up, other*self.Vp)

    __rmul__ = __mul__

# Check tangent vectors belong to same tangent space
def checkSameTangentSpace(xi, eta):
    if (xi.X != eta.X):
        print("Problem: tangent vectors from different tangent spaces. Operation not allowed")
        return False
    return True