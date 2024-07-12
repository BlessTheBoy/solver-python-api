import sympy as sp
import numpy as np


def muller_algorithm(equation: str, x0: float, x1: float, x2: float, variable: str = "x"):
  results = []
  fx = sp.sympify(equation)

  i = 0
  initialGuesses = [x0, x1, x2]

  while i < 10:  # Perform 4 iterations (as per the original code)
    points = [fx.subs(variable, value) for value in initialGuesses]

    h0 = initialGuesses[1] - initialGuesses[0]
    h1 = initialGuesses[2] - initialGuesses[1]
    d0 = (points[1] - points[0]) / (initialGuesses[1] - initialGuesses[0])
    d1 = (points[2] - points[1]) / (initialGuesses[2] - initialGuesses[1])


    a = (d1 - d0) / (h1 + h0)
    b = a*h1 + d1
    c = points[2]

    discriminant = sqrt(b**2 - 4*a*c)
    denominator = b - discriminant if b < 0 else b + discriminant
    x32 = (-2*c)/ denominator
    x3 = initialGuesses[2] + x32
    ea = abs((x32 / x3) * 100)

    result = {
        "iteration": i+1,
        "x0": initialGuesses[0],
        "x1": initialGuesses[1],
        "x2": initialGuesses[2],
        "xr": x3,
        "ea": ea
    }
    results.append(result)
    print(result, i)
    i+=1

    initialGuesses=[initialGuesses[1], initialGuesses[2], x3]

    if ea < 0.0001:
        break

  return results




def trapezoidal_rule(f: str, a: float, b: float, n: int):
  fx = sp.sympify(f)
  # fx = exp(x*tan(4*x))
  h = (b - a) / n
  result = 0.5 * (fx.subs("x", a).evalf() + fx.subs("x", b).evalf())
  for i in range(1, n):
      x = a + i * h
      result += fx.subs("x", x).evalf()
      # print("result", result)
  result *= h
  return result


# fx = sp.sympify("exp(x*tan(4*x))")
# print("fx", fx.subs("x", 2).evalf())


# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 1))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 2))
# print(trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 10))
# print("10 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 10))
# print("20 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 20))
# print("100 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 100))
# print("1000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 1000))
# print("3000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 3000))
# print("4000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 4000))
# print("5000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 5000))
# print("5000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 6000))
# print("5000 => ", trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 7000))
# print(trapezoidal_rule("exp(x*tan(4*x))", 1, 2, 100))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 4))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 5))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 6))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 7))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 8))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 9))
# print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 10))

# validate n as integer on frontend.











equation = "y2 + 0.01*(20-y)"
y_0 = 40
y_L = 200
L=10
h=2



def solve_differential_equation(equation: str, y_0: float, y_L: float, x_0: float, x_L: float, h: float):
  L= x_L - x_0
  n=int(L/h)

  x, y, p, q, r = sp.symbols("x y p q r")
  expr = sp.sympify(equation)

  # discretize the equation
  fir = sp.sympify("(p + r)/2*h")
  sec = sp.sympify("(p -2*q + r)/h**2")
  disc = expr.subs([("y1", fir), ("y2", sec), ("y", q), ("h", h), ("x", x)])

  # print(disc) 
  normalized_const = 1 / min(sp.diff(disc, p), sp.diff(disc, r))
  const_fact = (disc.subs([(p, 0), (q, 0), (r, 0)]) + sp.diff(disc, x)) * normalized_const
  disc_eqn = str(sp.simplify(disc * normalized_const).evalf() - const_fact) + " = " + str(-const_fact)
  print(disc_eqn)


  # Iterate through the equation to create the system of equations
  equations = []
  for i in range(1, n):
    y1 = sp.symbols(f"y{i-1}")
    y2 = sp.symbols(f"y{i}")
    y3 = sp.symbols(f"y{i+1}")
    equation_discretized = disc.subs([(p, y1), (q, y2), (r, y3), (x, x_0 + (i-1)*h)]).subs("y0", y_0).subs(f"y{n}", y_L)
    # print(equation_discretized)
    equations.append(equation_discretized)

  # Convert the system of equations to matrix form
  matrix = sp.zeros(n-1)
  for i, eq in enumerate(equations):
    for j, var in enumerate(eq.free_symbols):
      index = int(var.name[1:]) - 1
      matrix[i, index] = sp.diff(eq, var)


  constant_matrix = sp.zeros(n-1, 1)
  for i in range(n-1):
    v = disc
    if i == 0:
      v = disc.subs(p, y_0)
    elif i == n-2:
      v = disc.subs(r, y_L)
    constant_matrix[i, 0] = -v.subs([(p, 0), (q, 0), (r, 0), (x, x_0 + i*h)])

  sol = np.linalg.solve(np.array(matrix, dtype=np.float64), np.array(constant_matrix, dtype=np.float64))

  return {
    "disc_eqn": disc_eqn,
    "matrix": matrix,
    "constant_matrix": constant_matrix,
    "solution": sol
  }



solve_differential_equation("y2 + 0.01*(20-y)", 40, 200, 0, 10, 2)