#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# Define the function and its gradient
def f(x):
    return np.sqrt(1 + 2*x*x)

def df(x):
    return 4*x / np.sqrt(2*x**2 + 1)

def d2f(x):
    return 8*x**2 / (2*x*x + 1)*(np.sqrt(1 + 2*x*x))

# Define backtracking algorithm
def backtracking(x, pk, beta, c):
    alpha = 1
    while f(x + alpha*pk) > f(x) + c*alpha*np.dot(df(x), pk):
        alpha *= beta
    return alpha

# Damped Newton's Method
def damped_newton(x0, beta, c, epsilon):
    x = x0
    k = 0
    while abs(df(x)) > epsilon:
        pk = -df(x) / d2f(x)
        alpha = backtracking(x, pk, beta, c)
        x = x + alpha * pk
        k += 1
    return round(x, 9), k

# Set initial values
x0 = 1
beta = 0.9
c = 0.1
epsilon = 1e-4

# Apply Damped Newton's method
final_point, num_iterations = damped_newton(x0, beta, c, epsilon)
print("Final point:", final_point)
print("Number of iterations:", num_iterations)


# In[ ]:





# In[ ]:




