import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline

# 1. Datos de entrada
puntos = [
    (5,8), (5,14), (4,19), (3,22), (2,25), (1,28), (3,29), (4,28), (5,25), (6,23), 
    (8,21), (8,24), (8,27), (8,30), (8,34), (9,35), (10,35), (11,33), (11,29), 
    (12,26), (13,23), (14,29), (14,32), (14,36), (15,37), (16,36), (16,32), 
    (16,28), (16,23), (17,25), (18,28), (18,31), (18,34), (19,35), (20,34), 
    (20,30), (20,26), (19,32), (19,19), (20,16), (23,20), (25,22), (27,22), 
    (28,21), (27,20), (25,16), (23,12), (21,8), (18,5)
]

x_pts = np.array([p[0] for p in puntos])
y_pts = np.array([p[1] for p in puntos])
t_pts = np.arange(len(puntos)) # Parámetro t (índice del punto)
t_fino = np.linspace(0, len(puntos) - 1, 1000)

# --- MÉTODO 1: LAGRANGE (FORMA BARICÉNTRICA) ---
poly_lag_x = BarycentricInterpolator(t_pts, x_pts)
poly_lag_y = BarycentricInterpolator(t_pts, y_pts)
x_lag, y_lag = poly_lag_x(t_fino), poly_lag_y(t_fino)

# --- MÉTODO 2: NEWTON (DIFERENCIAS DIVIDIDAS) ---
def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i,j] = (coef[i+1,j-1] - coef[i,j-1]) / (x[i+j] - x[i])
    return coef[0,:]

def eval_newton(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n-k] + (x - x_data[n-k])*p
    return p

c_newton_x = divided_diff(t_pts, x_pts)
c_newton_y = divided_diff(t_pts, y_pts)
x_newton, y_newton = eval_newton(c_newton_x, t_pts, t_fino), eval_newton(c_newton_y, t_pts, t_fino)

# --- MÉTODO 3: MATRICIAL (VANDERMONDE) ---
# Resolvemos V * c = y. Nota: Muy inestable para grado 48
vander = np.vander(t_pts, increasing=True)
c_mat_x = np.linalg.solve(vander, x_pts)
c_mat_y = np.linalg.solve(vander, y_pts)

def eval_poly(c, t):
    return sum(coeff * (t**i) for i, coeff in enumerate(c))

x_mat, y_mat = eval_poly(c_mat_x, t_fino), eval_poly(c_mat_y, t_fino)

# --- MÉTODO 4: SPLINES CÚBICOS ---
cs_x = CubicSpline(t_pts, x_pts)
cs_y = CubicSpline(t_pts, y_pts)
x_spline, y_spline = cs_x(t_fino), cs_y(t_fino)

# --- GRAFICACIÓN ---
metodos = [
    ("1. Lagrange", x_lag, y_lag),
    ("2. Newton", x_newton, y_newton),
    ("3. Matricial", x_mat, y_mat),
    ("4. Splines Cúbicos", x_spline, y_spline)
]

fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for i, (nombre, x_c, y_c) in enumerate(metodos):
    axs[i].plot(x_c, y_c, 'r-', label='Interpolación')
    axs[i].scatter(x_pts, y_pts, color='blue', s=10, label='Puntos originales')
    axs[i].set_title(nombre)
    axs[i].set_xlim(min(x_pts)-2, max(x_pts)+2)
    axs[i].set_ylim(min(y_pts)-2, max(y_pts)+2)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend()

plt.tight_layout()
plt.show()
