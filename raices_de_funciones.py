import sympy as sp

def resolver_metodos():
    x = sp.symbols('x')
    
    print(" SOLUCIONADOR NUMERICO ")
    print("Ejemplo de funcion: x*cos(x) - 1")
    func_input = input("Ingrese la funcion f(x): ")

    try:
        expr = sp.sympify(func_input)
        f_lambda = sp.lambdify(x, expr, 'math')
        df_expr = sp.diff(expr, x)
        df_lambda = sp.lambdify(x, df_expr, 'math')
    except:
        print("Error en la sintaxis de la funcion.")
        return

    print("\nSeleccione el metodo:")
    print("1. Biseccion\n2. Newton-Raphson\n3. Secante\n4. Comparativo")
    opcion = input("Opcion: ")
    
    tol = float(input("Ingrese la tolerancia (ejemplo 1e-6): "))

    if opcion in ['1', '4']:
        print("\n BISECCIÓN ")
        a = float(input("Limite inferior (a): "))
        b = float(input("Limite superior (b): "))
        
        it = 0
        low, high = a, b
        if f_lambda(low) * f_lambda(high) >= 0:
            print("Error: No hay cambio de signo en el intervalo.")
        else:
            while True:
                it += 1
                c = (low + high) / 2
                error = abs(f_lambda(c))
                if error < tol:
                    print(f"Raiz: {c:.8f} | Iteraciones: {it} | Error: {error:.2e}")
                    break
                if f_lambda(low) * f_lambda(c) < 0: high = c
                else: low = c

    if opcion in ['2', '4']:
        print("\n NEWTON-RAPHSON ")
        xn = float(input("Punto inicial (x0): "))
        it = 0
        while True:
            it += 1
            f_val = f_lambda(xn)
            df_val = df_lambda(xn)
            if df_val == 0: break
            
            xn = xn - f_val / df_val
            error = abs(f_lambda(xn))
            if error < tol:
                print(f"Raiz: {xn:.8f} | Iteraciones: {it} | Error: {error:.2e}")
                break

    if opcion in ['3', '4']:
        print("\n SECANTE ")
        x0 = float(input("Punto x0: "))
        x1 = float(input("Punto x1: "))
        it = 0
        while True:
            it += 1
            f0, f1 = f_lambda(x0), f_lambda(x1)
            if f1 - f0 == 0: break
            
            x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
            error = abs(f_lambda(x_next))
            if error < tol:
                print(f"Raiz: {x_next:.8f} | Iteraciones: {it} | Error: {error:.2e}")
                break
            x0, x1 = x1, x_next

resolver_metodos()
