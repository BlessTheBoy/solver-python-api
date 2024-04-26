from sympy import *


def muller_algorithm(equation: str, x0: float, x1: float, x2: float, variable: str = "x"):
  results = []
  fx = sympify(equation)

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


print(muller_algorithm("x**3 - 13*x - 12", 4.5, 5.5, 5))