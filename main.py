from sympy import *
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
            jacobi_array = [[diff(sympify(item["eqn"]), v) for v in bodyValues.variables] for item in bodyValues.eqns]
        else:
            jacobi_array = [[sympify(item[v]) for v in bodyValues.variables] for item in bodyValues.eqns]
        
        print(jacobi_array)

            
        f_array = [sympify(item["eqn"]) for item in bodyValues.eqns]
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
    fx = sympify(bodyValues.equation)

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

        initialGuesses=[initialGuesses[1], initialGuesses[2], x3]

        # if (i > 0 and ea > results[i]["ea"]):
        #     divergenceCount += 1

        i+=1

        if ea < bodyValues.maxError or divergenceCount > 2:
            break
    res = {"results": results, "diverge": true if divergenceCount > 2 else false}
    jres = json.dumps(res)
    print(jres)
    # return {"results": results, "diverge": true if divergenceCount > 2 else false}
    return jres
