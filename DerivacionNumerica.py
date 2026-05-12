#EJERCICIO 1
import numpy as np
import pandas as pd

punto_evaluacion = 2.0
pasos_h = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
resultado_real = np.exp(punto_evaluacion)

calc_derivada_1 = []
calc_derivada_2 = []
err_relativo_1 = []
err_relativo_2 = []

for h in pasos_h:
    val_sup_6 = round(np.exp(punto_evaluacion + h), 6)
    val_inf_6 = round(np.exp(punto_evaluacion - h), 6)
    
    val_sup_8 = round(np.exp(punto_evaluacion + h), 8)
    val_central_8 = round(np.exp(punto_evaluacion), 8)
    val_inf_8 = round(np.exp(punto_evaluacion - h), 8)
    
    d1 = (val_sup_6 - val_inf_6) / (2 * h)
    d2 = (val_sup_8 - 2 * val_central_8 + val_inf_8) / (h**2)
    
    e1 = abs((d1 - resultado_real) / resultado_real) * 100
    e2 = abs((d2 - resultado_real) / resultado_real) * 100
    
    calc_derivada_1.append(round(d1, 6))
    calc_derivada_2.append(round(d2, 8))
    err_relativo_1.append(e1)
    err_relativo_2.append(e2)

df_resultados = pd.DataFrame({
    'h': pasos_h,
    "f'(x) aprox": calc_derivada_1,
    "f''(x) aprox": calc_derivada_2,
    "% Error D1": err_relativo_1,
    "% Error D2": err_relativo_2
})

print("REPORTE DE DIFERENCIAS FINITAS:")
print("=" * 80)
print(df_resultados.to_string(index=False))

#EJERCICIO 2

import numpy as np
import pandas as pd

nodos_x = np.array([1.5, 1.9, 2.1, 2.4, 2.6, 3.1])
nodos_y = np.exp(nodos_x)

df_datos = pd.DataFrame({'Abscisas (x)': nodos_x, 'Ordenadas f(x)': nodos_y})
print("VALORES DE LA FUNCIÓN:")
print("-" * 40)
print(df_datos.to_string(index=False))
print("\n")

punto_critico = 2.25

desviaciones = np.abs(nodos_x - punto_critico)
vecinos_indices = np.argsort(desviaciones)[:3]

x_subset = nodos_x[vecinos_indices]
y_subset = nodos_y[vecinos_indices]

print("NODOS SELECCIONADOS:")
print(f"X: {x_subset}")
print(f"Y: {y_subset}\n")

ajuste_cuadratico = np.polyfit(x_subset, y_subset, 2)
A, B, C = ajuste_cuadratico

derivada_primaria = 2 * A * punto_critico + B
derivada_secundaria = 2 * A

print("ESTIMACIÓN MEDIANTE POLINOMIO:")
print("-" * 40)
print(f"Primera Derivada en {punto_critico}: {derivada_primaria:.6f}")
print(f"Segunda Derivada en {punto_critico}: {derivada_secundaria:.6f}")

referencia_teorica = np.exp(punto_critico)
print(f"\nValor de control (analítico): {referencia_teorica:.6f}")
