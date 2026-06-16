import numpy as np
import matplotlib.pyplot as plt

mu_param = 1.5        
paso_h = 0.05          
tiempo = np.arange(0, 20 + paso_h, paso_h)  

posicion = np.zeros(len(tiempo))
velocidad = np.zeros(len(tiempo))

posicion[0] = 2.0     
velocidad[0] = 0.0     

def ecuacion_vanderpol(x_act, v_act, mu):
    """Define las derivadas del sistema (dx/dt y dv/dt)"""
    dxdt = v_act
    dvdt = mu * (1 - x_act**2) * v_act - x_act
    return dxdt, dvdt

for idx in range(len(tiempo) - 1):
    x_n, v_n = posicion[idx], velocidad[idx]
    
    dx1, dv1 = ecuacion_vanderpol(x_n, v_n, mu_param)
    
    dx2, dv2 = ecuacion_vanderpol(x_n + 0.5 * paso_h * dx1, 
                                  v_n + 0.5 * paso_h * dv1, mu_param)
    
    dx3, dv3 = ecuacion_vanderpol(x_n + 0.5 * paso_h * dx2, 
                                  v_n + 0.5 * paso_h * dv2, mu_param)
    
    dx4, dv4 = ecuacion_vanderpol(x_n + paso_h * dx3, 
                                  v_n + paso_h * dv3, mu_param)
    
    posicion[idx + 1] = x_n + (paso_h / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
    velocidad[idx + 1] = v_n + (paso_h / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)

print("=== VALORES INICIALES (PRIMEROS 10) ===")
print("Tiempo (s) | Posición (x) | Velocidad (v)")
for idx in range(10):
    print(f"{tiempo[idx]:.2f}       | {posicion[idx]:.4f}       | {velocidad[idx]:.4f}")

print("\n=== VALORES FINALES (ÚLTIMOS 5) ===")
print("Tiempo (s) | Posición (x) | Velocidad (v)")
for idx in range(len(tiempo) - 5, len(tiempo)):
    print(f"{tiempo[idx]:.2f}       | {posicion[idx]:.4f}       | {velocidad[idx]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.plot(tiempo, posicion, '-', label='Posición $x(t)$')
ax1.plot(tiempo, velocidad, '--', label='Velocidad $v(t)$')
ax1.set_title('Comportamiento Temporal')
ax1.set_xlabel('Tiempo (s)')
ax1.legend()
ax1.grid(True)


ax2.plot(posicion, velocidad, color='darkorchid')
ax2.plot(posicion[0], velocidad[0], 'go', label='Punto Inicial')
ax2.set_title('Espacio de Fases')
ax2.set_xlabel('Posición $x(t)$')
ax2.set_ylabel('Velocidad $v(t)$')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
