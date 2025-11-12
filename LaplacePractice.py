# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 17:20:36 2025

@author: lexma
"""

#Experimenting with Laplace transforms

from sympy import symbols, laplace_transform, inverse_laplace_transform, exp, sin, Eq, Function, dsolve, Derivative
t, s, a = symbols('t s a')
f = 1
F = laplace_transform(f, t, s)
print(F)

y = Function('y')
t = symbols('t')
ode = Eq(Derivative(y(t), t, t) + 3*Derivative(y(t), t) + 2*y(t), 0)
laplace_solution = dsolve(ode, ics={y(0): 1, Derivative(y(t), t).subs(t, 0): 0})
print(laplace_solution)