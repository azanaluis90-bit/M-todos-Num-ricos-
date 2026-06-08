import numpy as np
import sympy as sp

def ingresar_funcion():
    print("\n" + "="*65)
    print("   INTEGRACIÓN NUMÉRICA - EJERCICIO 1 (Lab)")
    print("="*65)
    print("\nEjemplos de funciones válidas:")
    print("  (1/sp.sqrt(2*sp.pi)) * sp.exp(-x**2/2)")
    print("  sp.cos(x**2)")
    print("  x**5 - 2*x**3 + 4")
    print("\nUse 'x' como variable y funciones de sympy (sp.sin, sp.exp, etc.)\n")

    expr_str = input("Ingrese la función f(x): ").strip()
    a = float(input("Límite inferior de integración (a): "))
    b = float(input("Límite superior de integración (b): "))

    x = sp.Symbol('x')
    try:
        expr = eval(expr_str, {"x": x, "sp": sp, "np": np})
    except Exception as e:
        raise ValueError(f"Error al parsear la función: {e}")

    f_sym = expr          # expresión simbólica
    f = sp.lambdify(x, expr, modules=["numpy", "sympy"])  # función numérica
    return f, f_sym, a, b, x

def es_polinomio(expr, x):
    """Devuelve (True, grado) si la expresión es polinómica, (False, None) si no."""
    try:
        p = sp.Poly(expr, x)
        return True, p.degree()
    except sp.PolynomialError:
        return False, None

def integral_exacta(f_sym, a, b, x):
    try:
        val = sp.integrate(f_sym, (x, a, b))
        return float(val.evalf())
    except Exception:
        return None

def trapecio_compuesto(f, a, b, N=12):
    h = (b - a) / N
    xs = [a + i * h for i in range(N + 1)]
    evaluaciones = N + 1
    suma = f(xs[0]) + f(xs[-1])
    for xi in xs[1:-1]:
        suma += 2 * f(xi)
    return h / 2 * suma, evaluaciones

def simpson_13_compuesto(f, a, b, N=12):
    if N % 2 != 0:
        N += 1  # N debe ser par
    h = (b - a) / N
    xs = [a + i * h for i in range(N + 1)]
    evaluaciones = N + 1
    suma = f(xs[0]) + f(xs[-1])
    for i in range(1, N):
        coef = 4 if i % 2 != 0 else 2
        suma += coef * f(xs[i])
    return h / 3 * suma, evaluaciones


def simpson_38_compuesto(f, a, b, N=12):
    if N % 3 != 0:
        N = N + (3 - N % 3)  # N debe ser múltiplo de 3
    h = (b - a) / N
    xs = [a + i * h for i in range(N + 1)]
    evaluaciones = N + 1
    suma = f(xs[0]) + f(xs[-1])
    for i in range(1, N):
        coef = 3 if i % 3 != 0 else 2
        suma += coef * f(xs[i])
    return 3 * h / 8 * suma, evaluaciones


def gauss_legendre_nodos_pesos(n):
    """
    Calcula algorítmicamente las raíces y pesos de Gauss-Legendre de n puntos
    usando el método de Newton sobre los polinomios de Legendre.
    """
    # Aproximación inicial de Gauss para las raíces
    raices = np.zeros(n)
    pesos  = np.zeros(n)

    for i in range((n + 1) // 2):
        # Aproximación inicial (fórmula de Gauss)
        xi = np.cos(np.pi * (i + 0.75) / (n + 0.5))

        # Iteración de Newton
        for _ in range(100):
            p0, p1 = 1.0, xi
            for k in range(2, n + 1):
                p0, p1 = p1, ((2*k - 1)*xi*p1 - (k - 1)*p0) / k
            # p1 = P_n(xi),  dp = derivada
            dp = n * (p0 - xi * p1) / (1 - xi**2)
            delta = p1 / dp
            xi -= delta
            if abs(delta) < 1e-15:
                break

        raices[i]         =  xi
        raices[n - 1 - i] = -xi
        w = 2.0 / ((1 - xi**2) * dp**2)
        pesos[i]         = w
        pesos[n - 1 - i] = w

    return raices, pesos


def gauss_legendre(f, a, b, n=5):
    """Cuadratura de Gauss-Legendre de n nodos en [a, b]."""
    raices, pesos = gauss_legendre_nodos_pesos(n)
    # Cambio de variable: t ∈ [-1,1] → x ∈ [a,b]
    factor = (b - a) / 2.0
    media  = (a + b) / 2.0
    xs = media + factor * raices
    resultado = factor * sum(w * f(x) for w, x in zip(pesos, xs))
    evaluaciones = n
    return resultado, evaluaciones


def elegir_n_optimo_gauss(f_sym, x):
    """
    Si la función es polinómica de grado d, Gauss-Legendre con n nodos
    es exacta para polinomios de grado ≤ 2n-1.
    Devuelve el n mínimo tal que 2n-1 ≥ d.
    """
    es_poli, grado = es_polinomio(f_sym, x)
    if es_poli:
        n_opt = int(np.ceil((grado + 1) / 2))
        return n_opt, grado
    return None, None

def imprimir_tabla(resultados, exacto):
    col_metodo = 26
    col_puntos = 12
    col_aprox  = 20
    col_error  = 18

    print("\n" + "─"*80)
    print(f"{'Método':<{col_metodo}}| {'Puntos F(x)':^{col_puntos}}| {'Aproximación':^{col_aprox}}| {'Error Absoluto':^{col_error}}")
    print("─"*80)

    for nombre, aprox, evals in resultados:
        if exacto is not None:
            error = abs(aprox - exacto)
            error_str = f"{error:.6e}"
        else:
            error_str = "N/A (sin exacto)"
        print(f"{nombre:<{col_metodo}}| {evals:^{col_puntos}}| {aprox:^{col_aprox}.10f}| {error_str:^{col_error}}")

    print("─"*80)
    if exacto is not None:
        print(f"  Valor exacto (simbólico): {exacto:.10f}")
    print()

def main():
    # 1) Ingreso de función
    f, f_sym, a, b, x = ingresar_funcion()

    # 2) Valor exacto
    exacto = integral_exacta(f_sym, a, b, x)

    # 3) Detección de polinomio para Gauss
    es_poli, grado = es_polinomio(f_sym, x)
    N = 12  # subintervalos para métodos compuestos

    if es_poli:
        print(f"\n  ✔ Función polinómica detectada (grado {grado}).")
        n_gl_opt, _ = elegir_n_optimo_gauss(f_sym, x)
        print(f"  ✔ Gauss-Legendre usará n = {n_gl_opt} nodos (exactitud garantizada).")
        n_gl = n_gl_opt
    else:
        print("\n  ✗ Función no polinómica. Gauss-Legendre usará n = 5 nodos.")
        n_gl = 5

    # 4) Aplicar métodos
    trap,   ev_trap  = trapecio_compuesto(f, a, b, N)
    s13,    ev_s13   = simpson_13_compuesto(f, a, b, N)
    s38,    ev_s38   = simpson_38_compuesto(f, a, b, N)
    gl,     ev_gl    = gauss_legendre(f, a, b, n_gl)

    # 5) Tabla de resultados
    resultados = [
        ("Trapecio Compuesto",    trap, ev_trap),
        ("Simpson 1/3 Compuesto", s13,  ev_s13),
        ("Simpson 3/8 Compuesto", s38,  ev_s38),
        (f"Gauss-Legendre (n={n_gl})", gl, ev_gl),
    ]

    print(f"\n  Función: f(x) = {f_sym}")
    print(f"  Intervalo: [{a}, {b}]")
    imprimir_tabla(resultados, exacto)

    # 6) Análisis comparativo
    if exacto is not None:
        metodos_names = ["Trapecio", "Simpson 1/3", "Simpson 3/8", f"Gauss-Legendre (n={n_gl})"]
        errores = [abs(trap - exacto), abs(s13 - exacto), abs(s38 - exacto), abs(gl - exacto)]
        mejor = metodos_names[np.argmin(errores)]
        evals_lista = [ev_trap, ev_s13, ev_s38, ev_gl]

        print("  ─── Análisis de Eficiencia ───────────────────────────────────")
        print(f"  Newton-Cotes usan {ev_trap} evaluaciones de f(x).")
        print(f"  Gauss-Legendre usó solo {ev_gl} evaluaciones.")
        print(f"  Método más preciso: {mejor} (error = {min(errores):.4e})")

        if min(errores) == errores[-1] or errores[-1] < 1e-10:
            print("\n  Gauss-Legendre obtuvo la mayor precisión con menos evaluaciones,")
            print("  demostrando su eficiencia computacional superior.")
        print("  ─"*32)

    # 7) Pregunta 3: límites del método
    print("\n  ─── Pregunta 3: Límites del Método ──────────────────────────")
    print("  P(x) = 7x^7 - 3x^4 + 2x  →  grado 7")
    print("  Gauss-Legendre con n nodos es exacto para grado ≤ 2n-1.")
    print("  Para grado 7:  2n-1 ≥ 7  →  n ≥ 4")
    print("  Con n = 4 nodos, Gauss-Legendre obtiene error = 0 exactamente.")
    print("  ─"*32 + "\n")


if __name__ == "__main__":
    main()
