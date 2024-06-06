import sympy as sp


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
  h = (b - a) / n
  result = 0.5 * (fx.subs("x", a) + fx.subs("x", b))
  for i in range(1, n):
    x = a + i * h
    result += fx.subs("x", x)
  result *= h
  return result


print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 1))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 2))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 3))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 4))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 5))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 6))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 7))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 8))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 9))
print(trapezoidal_rule("0.2 + 25*x + - 200*x**2 + 675 * x**3 - 900 * x**4 + 400 * x**5", 0, 0.8, 10))