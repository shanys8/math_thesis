import matlab.engine
import numpy as np
from numpy.linalg import matrix_power, norm
from scipy.linalg import sqrtm
import math


def calc_W_from_v(v):
    diag_v_pow_2 = matrix_power(np.diag(v), 2)
    coef = (12 * math.pow(n, 2)) / ((n - 1) * (n - 2) * (n - 3))
    # TODO A have also negative entries so sqrt is complex
    A = coef * (((n + 1) / n) * diag_v_pow_2 - (((n - 1) / math.pow(n, 2)) * np.trace(diag_v_pow_2)) * np.eye(n))
    U = math.sqrt(coef * ((2 * n - 2) / math.pow(n, 2))) * v
    # W = A - UU.T
    return A, U


def matlab_code_invocation(sqrt_A, identity, U, params):
    eng = matlab.engine.start_matlab()
    result = eng.Riemannian_lowrank_riccati_copy(matlab.double(sqrt_A.tolist()),
                                                 matlab.double(identity.tolist()),
                                                 matlab.double(list(U.T)), params)
    print(result)
    eng.quit()


# create ICA W as input to Ricatti algorithm
n = 100
X = np.random.rand(n, n)
u = np.random.rand(n)
identity = np.eye(n)

v = np.dot(X, u)
# v = v / norm(v)

A, U = calc_W_from_v(v)
A = A + np.abs(np.min(A)) * np.eye(n)  ## just for handleing real numbers
sqrt_A = sqrtm(A)
# TODO we handled the case when W = A+UUt (not minus)
W = A - np.outer(U, U)


params = {'rmax': 4, 'tol_rel': 1e-6, 'tolgradnorm': 1e-14, 'maxiter': 100, 'maxinner': 30, 'verbosity': 1}


matlab_code_invocation(sqrt_A, identity, U, params)

