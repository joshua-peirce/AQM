import numpy as np
import math

PI_3_2 = math.pi ** 1.5
N = 6 #number of terms used in expansion

ALPHA_LEARNING_RATE = 0.1
COEFF_LEARNING_RATE = 0.1
DELTA_A = 0.001
DELTA_C = 0.001

def calc_overlap(ap, aq):
    return PI_3_2 / ((ap + aq) ** 1.5)

def calc_KE(ap, aq):
    return 3 * PI_3_2 * (ap * aq) / (ap + aq) ** 2.5

def calc_PE(ap, aq):
    return -2 * math.pi / (ap + aq)

def calc_single_E(ap, aq):
    return calc_KE(ap, aq) + calc_PE(ap, aq)

def calc_composite_E(c, a, print_overlap=False):
     #E = \sum_i \sum_j C_i C_j <i | H | j> / <i | j>
     E = 0
     o = 0
     for i in range(N):
         for j in range(N):
             E += c[i] * c[j] * calc_single_E(a[i], a[j])
             o += c[i] * c[j] * calc_overlap(a[i], a[j])
     if print_overlap:
         print("Normalization:", o)
     return E / o

def calc_norm(c, a):
    o = 0
    for i in range(N):
        for j in range(N):
            o += c[i] * c[j] * calc_overlap(a[i], a[j])
    return o

def step(c, a):
    #Vary the parameters to see the direction of steepest descent.
    #Start with alphas and then move on to coefficients
    E_0 = calc_composite_E(c, a)
    #print(E_0)
    #Computing alphas
    gradients_a = []
    gradients_c = []
    for i in range(N):
        a[i] += DELTA_A
        gradients_a.append((calc_composite_E(c, a) - E_0) / DELTA_A)
        a[i] -= DELTA_A
    for i in range(N):
        c[i] += DELTA_C
        gradients_c.append((calc_composite_E(c, a) - E_0) / DELTA_C)
        c[i] -= DELTA_C
    a -= gradients_a * a # scaled by the alpha values so they never go negative
    c -= gradients_c
    return c, a

"""start_alphas = np.random.rand(N)#np.array([13.00773, 1.962079, 0.444529, 0.1219492], dtype=np.float64)
start_coeffs = np.random.rand(N)#np.array([1, 1, 1, 1, dtype=np.float64)
c = start_coeffs
a = start_alphas"""
c = np.array([0.82756178, 0.83292796, 0.32551525, 0.53784354, 0.25101611, 0.14072509])
a = np.array([0.76627919, 0.57237301, 0.77900288, 0.09414339, 0.98226739, 0.87722718])
for i in range(10 ** 4):
    c, a = step(c, a)
    if i % (10 ** 3) == 0:
        print(c, a, calc_composite_E(c, a))
print("Normalizing:")
c /= math.sqrt(calc_norm(c, a))
print(c, a, calc_composite_E(c, a, print_overlap=True))
