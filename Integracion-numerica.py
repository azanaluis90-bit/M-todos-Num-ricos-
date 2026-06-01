import numpy as np
from scipy.interpolate import CubicSpline

V = np.array([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4])
P = np.array([300.2, 242.1, 201.5, 169.8, 151, 134.3, 120.8, 107.9, 99.1, 93.4, 86.2, 79.5, 75.1])

n = 12
h = 0.25
V_min = 1.0
V_max = 4.0

def trapecio_compuesto(y, h):
    suma = y[0] + y[-1]
    for i in range(1, len(y) - 1):
        suma += 2 * y[i]
    return (h / 2) * suma

def simpson_13_compuesto(y, h):
    suma = y[0] + y[-1]
    for i in range(1, len(y) - 1):
        if i % 2 != 0:
            suma += 4 * y[i]
        else:
            suma += 2 * y[i]
    return (h / 3) * suma

def simpson_38_compuesto(y, h):
    suma = y[0] + y[-1]
    for i in range(1, len(y) - 1):
        if i % 3 == 0:
            suma += 2 * y[i]
        else:
            suma += 3 * y[i]
    return (3 * h / 8) * suma

P_interp = CubicSpline(V, P)

def gauss_legendre_3puntos(f, a, b):
    nodos_estandar = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    pesos_estandar = np.array([5/9, 8/9, 5/9])

    nodos_transformados = ((b - a) / 2) * nodos_estandar + ((b + a) / 2)
    pesos_transformados = ((b - a) / 2) * pesos_estandar
    
    integral = np.sum(pesos_transformados * f(nodos_transformados))
    return integral

w_trapecio = trapecio_compuesto(P, h)
w_simp13 = simpson_13_compuesto(P, h)
w_simp38 = simpson_38_compuesto(P, h)
w_gauss = gauss_legendre_3puntos(P_interp, V_min, V_max)

valor_referencia = w_gauss

error_trap = abs((w_trapecio - valor_referencia) / valor_referencia) * 100
error_s13 = abs((w_simp13 - valor_referencia) / valor_referencia) * 100
error_s38 = abs((w_simp38 - valor_referencia) / valor_referencia) * 100
error_gauss = 0.0 

print(f"{'MÉTODO':<18} | {'VALOR APROXIMADO':<16} | {'ERROR RELATIVO %':<16}")
print("-" * 58)
print(f"{'Trapecio':<18} | {w_trapecio:<16.4f} | {error_trap:<16.4f}%")
print(f"{'Simpson 1/3':<18} | {w_simp13:<16.4f} | {error_s13:<16.4f}%")
print(f"{'Simpson 3/8':<18} | {w_simp38:<16.4f} | {error_s38:<16.4f}%")
print(f"{'Gauss-Legendre':<18} | {w_gauss:<16.4f} | {error_gauss:<16.4f}% (Ref)")
