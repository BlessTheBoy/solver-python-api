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
            "y": float(yi),
            "d": float(f)
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
            "y": float(yi),
            "yi": float(f),
            "yi1": float(yi1),
            "d": float(corrector)
        }
        results.append(result)
        f = fx.subs("x", xi).subs("y", yi)
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
            "y": float(yi),
            "d": float(corrector)
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
            "y": float(y),
            "k1": float(k1),
            "k2": float(k2),
        }
        results.append(result)
    return results
    



