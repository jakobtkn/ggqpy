from typing import Callable

class Interval:
    def __init__(self, start: float, end: float) -> None:
        if (start > end):
            raise Exception("end must be greater than start") 
        self.a = start
        self.b = end
        return
    
    def __repr__(self):
        return (self.a,self.b)
    
    def __str__(self):
        return "(" + str(self.a) + "," +str(self.b) + ")"

class FunctionFamily:
    I = None
    functions = None
    def __init__(self, I: Interval, functions: list[Callable[[float],float]], integrals = None) -> None:
        self.I = I
        self.functions = functions
        self.analytic_integrals = integrals
        return