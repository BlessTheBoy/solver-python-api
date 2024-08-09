from math import sqrt
import sympy as sp
import numpy as np

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Annotated
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool

class StoppingCriteria(BaseModel):
    max_iterations: int
    max_error: float
class BodyType(BaseModel):
    auto_differentiate: bool
    variables: List[str]
    eqns: List[Dict[str, str]]
    initial_values: Dict[str, float]
    stopping_criteria: StoppingCriteria

# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
# class Result(BaseModel):
#             results: List[{
#                  iteration: int,
#             values: Dict[str, float],
#             errors: Dict[str, float]
#             }]

# @app.put("/systems-of-nonlinear-equations/newton-raphson")
# def update_item(autoDifferentiate: int, item: Parameters):
#     return {"item_name": item.name, "item_id": item_id}
    

@app.get("/")
def home():
    return {"msg": "Welcome in my api"}

@app.post("/systems-of-nonlinear-equations/newton-raphson")
async def jacobi_iteration(bodyValues: BodyType):
    try:
        if bodyValues.auto_differentiate:
            jacobi_array = [[sp.diff(sp.sympify(item["eqn"]), v) for v in bodyValues.variables] for item in bodyValues.eqns]
        else:
            jacobi_array = [[sp.sympify(item[v]) for v in bodyValues.variables] for item in bodyValues.eqns]
        
        print(jacobi_array)

            
        f_array = [sp.sympify(item["eqn"]) for item in bodyValues.eqns]
    except:
        raise HTTPException(status_code=404, detail="Invalid Equation")
        
    prev_values = np.array([bodyValues.initial_values[key] for key in bodyValues.variables])

    results = []

    i = 0
    while i < bodyValues.stopping_criteria.max_iterations:  # Perform 4 iterations (as per the original code)
        jacobian = np.array([[float(item.evalf(subs={key: prev_values[ind] for ind, key in enumerate(bodyValues.variables)})) for item in row] for row in jacobi_array])
        f = np.array([float(item.evalf(subs={key: prev_values[ind] for ind, key in enumerate(bodyValues.variables)}) * -1) for item in f_array]).T

        delta = np.linalg.inv(jacobian).dot(f)
        values = delta + prev_values

        relative_absolute_errors = {}
        for ind, var in enumerate(bodyValues.variables):
            if prev_values[ind] != 0:
                relative_absolute_errors[var] = abs(delta[ind] / prev_values[ind]) * 100
            else:
                relative_absolute_errors[var] = None

        prev_values = values.copy()

        result = {
            "iteration": i+1,
            "values": {key:values[ind] for ind, key in enumerate(bodyValues.variables)},
            "errors": relative_absolute_errors
        }
        results.append(result)
        print(result, i)
        i+=1

        c = [relative_absolute_errors[x] < bodyValues.stopping_criteria.max_error for x in bodyValues.variables]
        print (c)

        if all([relative_absolute_errors[x] < bodyValues.stopping_criteria.max_error for x in bodyValues.variables]):
            break

    return {"results": results}


class MullerBodyType(BaseModel):
    equation: str
    x0: float
    x1: float
    x2: float
    maxIterations: int
    maxError: float

@app.post("/roots-of-polynomials/muller")
async def muller_method(bodyValues: MullerBodyType):
    results = []
    divergenceCount = 0
    fx = sp.sympify(bodyValues.equation)

    i = 0
    initialGuesses = [bodyValues.x0, bodyValues.x1, bodyValues.x2]

    while i < bodyValues.maxIterations:  # Perform 4 iterations (as per the original code)
        points = [fx.subs("x", value) for value in initialGuesses]

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
        x3 = initialGuesses[2] + float(x32)
        ea = abs((x32 / x3) * 100)

        result = {
            "iteration": i+1,
            "x0": str(initialGuesses[0]),
            "x1": str(initialGuesses[1]),
            "x2": str(initialGuesses[2]),
            "xr": str(x3),
            "ea": str(ea)
        }
        results.append(result)
        print(result, i)

        initialGuesses=[initialGuesses[1], initialGuesses[2], x3]

        # if (i > 0 and ea > results[i]["ea"]):
        #     divergenceCount += 1

        i+=1

        if ea < bodyValues.maxError or divergenceCount > 2:
            break
    res = {"results": results, "diverge": True if divergenceCount > 2 else False}
    # jres = json.dumps(res, default=str)
    # print(jres)
    # return {"results": results, "diverge": true if divergenceCount > 2 else false}
    return res




class TrapezoidalBodyType(BaseModel):
    dataType: str
    equation: str
    a: float
    b: float
    n: int

@app.post("/newton-cotes/trapezoidal")
async def trapezoidal_rule(bodyValues: TrapezoidalBodyType):
    # dataType = bodyValues.dataType
    equation = bodyValues.equation
    a = bodyValues.a
    b = bodyValues.b
    n = bodyValues.n
    fx = sp.sympify(equation)
    h = (b - a) / n
    result = 0.5 * (fx.subs("x", a) + fx.subs("x", b))
    for i in range(1, n):
        x = a + i * h
        result += fx.subs("x", x)
        # print("result", result)
    result *= h
    return float(result)



@app.post("/newton-cotes/simpson1")
async def simpson1_rule(bodyValues: TrapezoidalBodyType):
    # dataType = bodyValues.dataType
    equation = bodyValues.equation
    a = bodyValues.a
    b = bodyValues.b
    n = bodyValues.n
    fx = sp.sympify(equation)
    h = (b - a) / n
    result = (fx.subs("x", a) + fx.subs("x", b))
    for i in range(1, n, 2):
        x = a + i * h
        result += (4 * fx.subs("x", x))
    for i in range(2, n, 2):
        x = a + i * h
        result += (2 * fx.subs("x", x))
    result *= (h/3)
    return float(result)


@app.post("/newton-cotes/simpson3")
async def simpson3_rule(bodyValues: TrapezoidalBodyType):
    # dataType = bodyValues.dataType
    equation = bodyValues.equation
    a = bodyValues.a
    b = bodyValues.b
    n = bodyValues.n
    fx = sp.sympify(equation)
    h = (b - a) / n
    result = (fx.subs("x", a) + fx.subs("x", b))
    for i in range(1, n):
        x = a + i * h
        if i%3 == 0 :
            result += (2 * fx.subs("x", x))
        else: 
            result += (3 * fx.subs("x", x))
    result *= (h*(3/8))
    return float(result)


@app.post("/newton-cotes/boole")
async def boole(bodyValues: TrapezoidalBodyType):
    # dataType = bodyValues.dataType
    equation = bodyValues.equation
    a = bodyValues.a
    b = bodyValues.b
    n = bodyValues.n
    fx = sp.sympify(equation)
    h = (b - a) / n
    result = 7 * (fx.subs("x", a) + fx.subs("x", b))
    for i in range(1, n):
        x = a + i * h
        if i%4 == 0 :
            result += (14 * fx.subs("x", x))
        elif i%2 == 0:
            result += (12 * fx.subs("x", x))
        else: 
            result += (32 * fx.subs("x", x))
    result *= (h*(4/90))
    return float(result)




class HadfBodyType(BaseModel):
    dataType: str
    equation: str
    order: int
    type: str
    x: float
    h: float

@app.post("/differentiation/hadf")
async def high_accuracy_differential_formula(bodyValues: HadfBodyType):
    equation = bodyValues.equation
    order = bodyValues.order
    type = bodyValues.type
    x = bodyValues.x
    h = bodyValues.h
    fx = sp.sympify(equation)
    if type == "forward":
        if order == 1:
            result = (-3*fx.subs("x", x) + 4*fx.subs("x", x+h) - fx.subs("x", x+2*h)) / (2*h)
        elif order == 2:
            result = (2*fx.subs("x", x) - 5*fx.subs("x", x+h) + 4*fx.subs("x", x+2*h) - fx.subs("x", x+3*h)) / (h**2)
        elif order == 3:
            result = (-11*fx.subs("x", x) + 18*fx.subs("x", x+h) - 9*fx.subs("x", x+2*h) + 2*fx.subs("x", x+3*h)) / (6*h)
        elif order == 4:
            result = (2*fx.subs("x", x) - 5*fx.subs("x", x+h) + 4*fx.subs("x", x+2*h) - fx.subs("x", x+3*h)) / (h**3)
    elif type == "backward":
        if order == 1:
            result = (3*fx.subs("x", x) - 4*fx.subs("x", x-h) + fx.subs("x", x-2*h)) / (2*h)
        elif order == 2:
            result = (fx.subs("x", x) - 2*fx.subs("x", x-h) + fx.subs("x", x-2*h)) / (h**2)
        elif order == 3:
            result = (11*fx.subs("x", x) - 18*fx.subs("x", x-h) + 9*fx.subs("x", x-2*h) - 2*fx.subs("x", x-3*h)) / (6*h)
        elif order == 4:
            result = (2*fx.subs("x", x) - 5*fx.subs("x", x-h) + 4*fx.subs("x", x-2*h) - fx.subs("x", x-3*h)) / (h**3)
    elif type == "central":
        if order == 1:
            result = (fx.subs("x", x+h) - fx.subs("x", x-h)) / (2*h)
        elif order == 2:
            result = (fx.subs("x", x+h) - 2*fx.subs("x", x) + fx.subs("x", x-h)) / (h**2)
        elif order == 3:
            result = (fx.subs("x", x+2*h) - 2*fx.subs("x", x+h) + 2*fx.subs("x", x-h) - fx.subs("x", x-2*h)) / (2*h**3)
        elif order == 4:
            result = (fx.subs("x", x+2*h) - 4*fx.subs("x", x+h) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h) + fx.subs("x", x-2*h)) / (h**4)
    return float(result)




class RichardsonBodyType(BaseModel):
    dataType: str
    equation: str
    order: int
    type: str
    x: float
    h1: float
    h2: float

@app.post("/differentiation/richardson")
async def richardson_extrapolation(bodyValues: RichardsonBodyType):
    equation = bodyValues.equation
    order = bodyValues.order
    type = bodyValues.type
    x = bodyValues.x
    h1 = bodyValues.h1
    h2 = bodyValues.h2
    fx = sp.sympify(equation)
    if type == "forward":
        if order == 1:
            D1 = (fx.subs("x", x+h1) - fx.subs("x", x)) / h1
            D2 = (fx.subs("x", x+h2) - fx.subs("x", x)) / h2
        elif order == 2:
            D1 = (fx.subs("x", x+h1) - 2*fx.subs("x", x) + fx.subs("x", x-h1)) / h1**2
            D2 = (fx.subs("x", x+h2) - 2*fx.subs("x", x) + fx.subs("x", x-h2)) / h2**2
        elif order == 3:
            D1 = (fx.subs("x", x+2*h1) - 2*fx.subs("x", x+h1) + 2*fx.subs("x", x-h1) - fx.subs("x", x-2*h1)) / (2*h1**3)
            D2 = (fx.subs("x", x+2*h2) - 2*fx.subs("x", x+h2) + 2*fx.subs("x", x-h2) - fx.subs("x", x-2*h2)) / (2*h2**3)
        elif order == 4:
            D1 = (fx.subs("x", x+2*h1) - 4*fx.subs("x", x+h1) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h1) + fx.subs("x", x-2*h1)) / h1**4
            D2 = (fx.subs("x", x+2*h2) - 4*fx.subs("x", x+h2) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h2) + fx.subs("x", x-2*h2)) / h2**4
    elif type == "backward":
        if order == 1:
            D1 = (fx.subs("x", x) - fx.subs("x", x-h1)) / h1
            D2 = (fx.subs("x", x) - fx.subs("x", x-h2)) / h2
        elif order == 2:
            D1 = (fx.subs("x", x) - 2*fx.subs("x", x) + fx.subs("x", x-h1)) / h1**2
            D2 = (fx.subs("x", x) - 2*fx.subs("x", x) + fx.subs("x", x-h2)) / h2**2
        elif order == 3:
            D1 = (fx.subs("x", x+2*h1) - 2*fx.subs("x", x+h1) + 2*fx.subs("x", x-h1) - fx.subs("x", x-2*h1)) / (2*h1**3)
            D2 = (fx.subs("x", x+2*h2) - 2*fx.subs("x", x+h2) + 2*fx.subs("x", x-h2) - fx.subs("x", x-2*h2)) / (2*h2**3)
        elif order == 4:
            D1 = (fx.subs("x", x+2*h1) - 4*fx.subs("x", x+h1) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h1) + fx.subs("x", x-2*h1)) / h1**4
            D2 = (fx.subs("x", x+2*h2) - 4*fx.subs("x", x+h2) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h2) + fx.subs("x", x-2*h2)) / h2**4
    elif type == "central":
        if order == 1:
            D1 = (fx.subs("x", x+h1) - fx.subs("x", x-h1)) / (2*h1)
            D2 = (fx.subs("x", x+h2) - fx.subs("x", x-h2)) / (2*h2)
        elif order == 2:
            D1 = (fx.subs("x", x+h1) - 2*fx.subs("x", x) + fx.subs("x", x-h1)) / h1**2
            D2 = (fx.subs("x", x+h2) - 2*fx.subs("x", x) + fx.subs("x", x-h2)) / h2**2
        elif order == 3:
            D1 = (fx.subs("x", x+2*h1) - 2*fx.subs("x", x+h1) + 2*fx.subs("x", x-h1) - fx.subs("x", x-2*h1)) / (2*h1**3)
            D2 = (fx.subs("x", x+2*h2) - 2*fx.subs("x", x+h2) + 2*fx.subs("x", x-h2) - fx.subs("x", x-2*h2)) / (2*h2**3)
        elif order == 4:
            D1 = (fx.subs("x", x+2*h1) - 4*fx.subs("x", x+h1) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h1) + fx.subs("x", x-2*h1)) / h1**4
            D2 = (fx.subs("x", x+2*h2) - 4*fx.subs("x", x+h2) + 6*fx.subs("x", x) - 4*fx.subs("x", x-h2) + fx.subs("x", x-2*h2)) / h2**4
    result = (4/3 * D2) - (1/3 * D1)
    return {"D": float(result), "D1": float(D1), "D2": float(D2)}



class eulerBodyType(BaseModel):
    equation: str
    x1: float
    x2: float
    y1: float
    h: float
    
class odeSysBodyType(BaseModel):
    equations: list[dict[str, str]]
    initialValues: list[float]
    x1: float
    x2: float
    h: float

class bvpBodyType(BaseModel):
    equation: str
    x_0: float
    x_L: float
    y_0: float
    y_L: float
    h: float

@app.post("/ode/euler")
async def euler_method(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    results = [{"iteration": 0, "x": float(x), "y": float(y), "d": float(f)}]
    for i in range(n):
        yi = y + h * f
        xi = x + h
        f = fx.subs("x", xi).subs("y", yi)
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": str(yi),
            "d": str(f)
        }
        results.append(result)
        x = xi
        y = yi
    return results


def substitute_values(expression, values):
    substituted_expression = expression
    for i, value in enumerate(values):
        substituted_expression = substituted_expression.subs(f"y{i+1}", value)
    return substituted_expression


@app.post("/sys-of-ode/euler")
async def system_of_ode_euler_method(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")

    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = initialValues
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    # print("f = ", f)
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        yi = [y[j] + h * f[j] for j in range(len(y))]
        print("yi = ", yi)
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))]
        }
        results.append(result)
        x = xi
        y = yi
    return results
    


@app.post("/ode/heun")
async def heun_method(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    results = [{"iteration": 0, "x": float(x), "y": float(y)}]
    for i in range(n):
        yi1 = fx.subs("x", x+h).subs("y", y+h*f)
        corrector = 0.5 * (f + yi1)
        yi = y + corrector * h
        xi = x + h
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": str(yi),
            "yi": str(f),
            "yi1": str(yi1),
            "d": str(corrector)
        }
        results.append(result)
        f = fx.subs("x", xi).subs("y", yi)
        x = xi
        y = yi
    return results


@app.post("/sys-of-ode/heun")
async def system_of_ode_heun_method(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")
    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = initialValues
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    # print("f = ", f)
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        yi1 = [substitute_values(fx[j].subs("x", x+h), y) for j in range(len(y))]
        corrector = [0.5 * (f[j] + yi1[j]) for j in range(len(y))]
        yi = [y[j] + corrector[j] * h for j in range(len(y))]
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))],
        }
        results.append(result)
        x = xi
        y = yi
    return results


@app.post("/ode/midpoint")
async def midpoint_euler(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    yi1 = y + (h/2) * f
    corrector = fx.subs("x", x+h/2).subs("y", yi1)
    results = [{"iteration": 0, "x": float(x), "y": float(y), "d": float(corrector)}]
    for i in range(n):
        yi = y + corrector * h
        xi = x + h
        f = fx.subs("x", xi).subs("y", yi)
        yi1 = yi + (h/2) * f
        corrector = fx.subs("x", xi+h/2).subs("y", yi1)
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": str(yi),
            "d": str(corrector)
        }
        results.append(result)
        x = xi
        y = yi
    return results

@app.post("/sys-of-ode/midpoint")
async def system_of_ode_midpoint_euler(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")
    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = initialValues
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    yi1 = [y[j] + (h/2) * f[j] for j in range(len(y))]
    corrector = [substitute_values(fx[i].subs("x", x+h/2), yi1) for i in range(len(y))]
    results = [{"iteration": 0, "x": float(x), "y": float(y), "d": float(corrector)}]
    for i in range(n):
        yi = [y[j] + corrector[j] * h for j in range(len(y))]
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        yi1 = [yi[j] + (h/2) * f[j] for j in range(len(y))]
        corrector = [substitute_values(fx[i].subs("x", xi+h/2), yi1) for i in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))],
        }
        results.append(result)
        x = xi
        y = yi
    return results
    



@app.post("/ode/ralston")
async def rk_ralston(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    results = [{"iteration": 0, "x": float(x), "y": float(y)}]
    for i in range(n):
        k1 = f
        k2 = fx.subs("x", x + (3/4)*h).subs("y", y + (3/4)*k1*h)
        y = y + (1/3)*k1*h + (2/3)*k2*h
        x = x + h
        f = fx.subs("x", x).subs("y", y)
        result = {
            "iteration": i+1,
            "x": float(x),
            "y": str(y),
            "k1": str(k1),
            "k2": str(k2),
        }
        results.append(result)
    return results


@app.post("/sys-of-ode/ralston")
async def system_of_ode_rk_ralston(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")
    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = initialValues
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        k1 = [f[j] for j in range(len(y))]
        k2 = [substitute_values(fx[j].subs("x", x + (3/4)*h), [y[k] + (3/4)*k1[k]*h for k in range(len(y))]) for j in range(len(y))]
        yi = [y[k] + (1/3)*k1[k]*h + (2/3)*k2[k]*h for k in range(len(y))]
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))],
        }
        results.append(result)
        x = xi
        y = yi
    return results

    

@app.post("/ode/rk3")
async def rk_3(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    results = [{"iteration": 0, "x": float(x), "y": float(y)}]
    for i in range(n):
        k1 = f
        k2 = fx.subs("x", x + (1/2)*h).subs("y", y + (1/2)*k1*h)
        k3 = fx.subs("x", x + h).subs("y", y - k1*h + 2*k2*h)
        y = y + (1/6)*h*(k1 + 4*k2 + k3)
        x = x + h
        f = fx.subs("x", x).subs("y", y)
        result = {
            "iteration": i+1,
            "x": float(x),
            "y": str(y),
            "k1": str(k1),
            "k2": str(k2),
            "k3": str(k3),
        }
        results.append(result)
    return results
    

@app.post("/sys-of-ode/rk3")
async def system_of_ode_rk_3(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")
    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = initialValues
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        k1 = [f[j] for j in range(len(y))]
        k2 = [substitute_values(fx[j].subs("x", x + (1/2)*h), [y[k] + (1/2)*k1[k]*h for k in range(len(y))]) for j in range(len(y))]
        k3 = [substitute_values(fx[j].subs("x", x + h), [y[k] - k1[k]*h + 2*k2[k]*h for k in range(len(y))]) for j in range(len(y))]
        yi = [y[k] + (1/6)*h*(k1[k] + 4*k2[k] + k3[k]) for k in range(len(y))]
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))],
        }
        results.append(result)
        x = xi
        y = yi
    return results
    


@app.post("/ode/rk4")
async def rk_4(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    print("===========> 1")
    fx = sp.sympify(equation)
    print("===========> 2")
    n = int((x2 - x1) / h)
    print("===========> 3")
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    print("===========> 4")
    results = [{"iteration": 0, "x": float(x), "y": float(y)}]
    for i in range(n):
        k1 = f
        k2 = fx.subs("x", x + (1/2)*h).subs("y", y + (1/2)*k1*h)
        k3 = fx.subs("x", x + (1/2)*h).subs("y", y + (1/2)*k2*h)
        k4 = fx.subs("x", x + h).subs("y", y + k3*h)
        y = y + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
        print("===========> 5")
        x = x + h
        f = fx.subs("x", x).subs("y", y)
        result = {
            "iteration": i+1,
            "x": float(x),
            "y": str(y),
            "k1": str(k1),
            "k2": str(k2),
            "k3": str(k3),
            "k4": str(k4),
        }
        # print("y: ", y)
        results.append(result)
    # print(results)
    return results


@app.post("/sys-of-ode/rk4")
async def system_of_ode_rk_4(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")

    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = initialValues
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    # print("f = ", f)
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        k1 = f
        k2 = [substitute_values(fx[j].subs("x", x + (1/2)*h), [y[j] + (1/2)*k1[j]*h for j in range(len(y))]) for j in range(len(y))]
        k3 = [substitute_values(fx[j].subs("x", x + (1/2)*h), [y[j] + (1/2)*k2[j]*h for j in range(len(y))]) for j in range(len(y))]
        k4 = [substitute_values(fx[j].subs("x", x + h), [y[j] + k3[j]*h for j in range(len(y))]) for j in range(len(y))]
        y = [y[j] + (1/6)*h*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) for j in range(len(y))]
        x = x + h
        f = [substitute_values(fx[j].subs("x", x), y) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(x),
            "y": [str(yi) for yi in y]
        }
        results.append(result)
    return results
    

@app.post("/ode/rk5")
async def rk5_butchers_method(bodyValues: eulerBodyType):
    equation = bodyValues.equation
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    y1 = bodyValues.y1
    h = bodyValues.h
    fx = sp.sympify(equation)
    n = int((x2 - x1) / h)
    x = x1
    y = y1
    f = fx.subs("x", x).subs("y", y)
    results = [{"iteration": 0, "x": float(x), "y": float(y)}]
    for i in range(n):
        k1 = f
        k2 = fx.subs("x", x + (1/4)*h).subs("y", y + (1/4)*k1*h)
        k3 = fx.subs("x", x + (1/4)*h).subs("y", y + (1/8)*k1*h + (1/8)*k2*h)
        k4 = fx.subs("x", x + (1/2)*h).subs("y", y - (1/2)*k2*h + k3*h)
        k5 = fx.subs("x", x + (3/4)*h).subs("y", y + (3/16)*k1*h + (9/16)*k4*h)
        k6 = fx.subs("x", x + h).subs("y", y - (3/7)*k1*h + (2/7)*k2*h + (12/7)*k3*h - (12/7)*k4*h + (8/7)*k5*h)
        y = y + (1/90)*h*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)
        x = x + h
        f = fx.subs("x", x).subs("y", y)
        result = {
            "iteration": i+1,
            "x": float(x),
            "y": str(y),
            "k1": str(k1),
            "k2": str(k2),
            "k3": str(k3),
            "k4": str(k4),
            "k5": str(k5),
            "k6": str(k6),
        }
        results.append(result)
    return results

    
@app.post("/sys-of-ode/rk5")
async def system_of_ode_rk5_butchers_method(bodyValues: odeSysBodyType):
    x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33, y34, y35, y36, y37, y38, y39, y40, y41, y42, y43, y44, y45, y46, y47, y48, y49, y50, y51, y52, y53, y54, y55, y56, y57, y58, y59, y60, y61, y62, y63, y64, y65, y66, y67, y68, y69, y70, y71, y72, y73, y74, y75, y76, y77, y78, y79, y80, y81, y82, y83, y84, y85, y86, y87, y88, y89, y90, y91, y92, y93, y94, y95, y96, y97, y98, y99, y100 = sp.symbols("x y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50 y51 y52 y53 y54 y55 y56 y57 y58 y59 y60 y61 y62 y63 y64 y65 y66 y67 y68 y69 y70 y71 y72 y73 y74 y75 y76 y77 y78 y79 y80 y81 y82 y83 y84 y85 y86 y87 y88 y89 y90 y91 y92 y93 y94 y95 y96 y97 y98 y99 y100")
    equations = bodyValues.equations
    initialValues = bodyValues.initialValues
    x1 = bodyValues.x1
    x2 = bodyValues.x2
    h = bodyValues.h
    fx = [sp.sympify(equation["equation"]) for equation in equations]
    n = int((x2 - x1) / h)
    x = x1
    y = initialValues
    f = [substitute_values(fx[i].subs("x", x), y) for i in range(len(y))]
    # print("f = ", f)
    results = [{"iteration": 0, "x": float(x), "y": [float(yi) for yi in y]}]
    for i in range(n):
        k1 = [f[j] for j in range(len(y))]
        k2 = [substitute_values(fx[j].subs("x", x + (1/4)*h), [y[j] + (1/4)*k1[j]*h for j in range(len(y))]) for j in range(len(y))]
        k3 = [substitute_values(fx[j].subs("x", x + (1/4)*h), [y[j] + (1/8)*k1[j]*h + (1/8)*k2[j]*h for j in range(len(y))]) for j in range(len(y))]
        k4 = [substitute_values(fx[j].subs("x", x + (1/2)*h), [y[j] - (1/2)*k2[j]*h + k3[j]*h for j in range(len(y))]) for j in range(len(y))]
        k5 = [substitute_values(fx[j].subs("x", x + (3/4)*h), [y[j] + (3/16)*k1[j]*h + (9/16)*k4[j]*h for j in range(len(y))]) for j in range(len(y))]
        k6 = [substitute_values(fx[j].subs("x", x + h), [y[j] - (3/7)*k1[j]*h + (2/7)*k2[j]*h + (12/7)*k3[j]*h - (12/7)*k4[j]*h + (8/7)*k5[j]*h for j in range(len(y))]) for j in range(len(y))]
        yi = [y[j] + (1/90)*h*(7*k1[j] + 32*k3[j] + 12*k4[j] + 32*k5[j] + 7*k6[j]) for j in range(len(y))]
        xi = x + h
        f = [substitute_values(fx[j].subs("x", xi), yi) for j in range(len(y))]
        result = {
            "iteration": i+1,
            "x": float(xi),
            "y": [str(yi[j]) for j in range(len(y))],
        }
        results.append(result)
        x = xi
        y = yi
    return results
   

    
@app.post("/bvp/finite-difference")
async def solve_differential_equation(bodyValues: bvpBodyType):
    equation = bodyValues.equation
    x_0 = bodyValues.x_0
    x_L = bodyValues.x_L
    y_0 = bodyValues.y_0
    y_L = bodyValues.y_L
    h = bodyValues.h


    L= x_L - x_0
    n=int(L/h)

    x, y, p, q, r = sp.symbols("x y p q r")
    expr = sp.sympify(equation)
    # l=5
    # y0 = 40
    # y5 = 200

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
        equations.append(equation_discretized * normalized_const)

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
        constant_matrix[i, 0] = -normalized_const * v.subs([(p, 0), (q, 0), (r, 0), (x, x_0 + i*h)])

    # print(matrix * normalized_const)
    # print(constant_matrix * normalized_const)

    sol = np.linalg.solve(np.array(matrix, dtype=np.float64), np.array(constant_matrix, dtype=np.float64))

    print(sol)

    results = {
        "disc_eqn": str(disc_eqn),
        "matrix": [[str(value) for value in row] for row in matrix.tolist()],
        "constant_matrix": [[str(value) for value in row] for row in constant_matrix.tolist()],
        "solution": [[str(value) for value in row] for row in sol.tolist()]
    }
    
    return results



class ellipticBodyType(BaseModel):
    nx: int
    ny: int
    l: float
    r: float
    t: float
    b: float
    max_iterations: int
    max_error: float
    over_relaxation: float

@app.post("/pde/finite-difference-elliptic")
async def partial_differential_equation_liebmann(bodyValues: ellipticBodyType):
    nx = bodyValues.nx
    ny = bodyValues.ny
    left = bodyValues.l
    right = bodyValues.r
    top = bodyValues.t
    bottom = bodyValues.b
    max_iterations = bodyValues.max_iterations
    abs_max_error = bodyValues.max_error
    over_relaxation = bodyValues.over_relaxation

    matrix = [[top] * (ny+2)]
    for i in range(1, nx+1):
        row = [left] + [0] * ny + [right]
        matrix.append(row)
    matrix.append([bottom] * (ny+2))

    results = [{"itr": 0, "matrix": matrix}]

    for iteration in range(max_iterations):
        new_matrix = [row.copy() for row in matrix]
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                new_value = (1 - over_relaxation) * new_matrix[i][j] + over_relaxation * (new_matrix[i-1][j] + new_matrix[i+1][j] + new_matrix[i][j-1] + new_matrix[i][j+1]) / 4
                new_matrix[i][j] = new_value
        
        max_error = max(abs((new_matrix[i][j] - matrix[i][j])/matrix[i][j]) * 100 if matrix[i][j] != 0 else 100 for i in range(1, nx+1) for j in range(1, ny+1))
        matrix = new_matrix
        results.append({"itr": iteration+1, "matrix": [[float(value) for value in row] for row in matrix], "abre": max_error})
        if max_error < abs_max_error:
            break
    
    return results




    




