#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import CMO_A2  # Assuming you have access to the CMO_A2 module

# Replace this with your last five digits of the SR Number
last_five_digits = 22888

# Obtain the matrix Q using the oracle
Q = CMO_A2.oracle4(last_five_digits)

rows, cols = (5, 5)
e = [[0]*cols]*rows
# e([[ 0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.]])
# Define the standard basis vectors
e[0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
e[1] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
e[2] = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
e[3] = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
e[4] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

# Initialize variables
u = [e0]
x = np.zeros(5)  # x0 = 0

# Use the Conjugate Directions Method to find u0, u1, ..., u4
for i in range(1, 5):
    numerator = e[i]
    denominator = 0
    for j in range(i):
        numerator -= (np.dot(e[i], np.dot(Q, u[j])) / np.dot(u[j], np.dot(Q, u[j]))) * u[j]
        denominator += (np.dot(u[j], np.dot(Q, u[j])))
    ui = numerator / denominator
    u.append(ui)

# Define b based on the last five digits of your SR Number

b = np.array([2, 2, 8, 8, 8])

# Use the Conjugate Directions Method to minimize the function
for i in range(5):
    alpha = (np.dot(u[i], b) / np.dot(u[i], np.dot(Q, u[i])))
    x = x + alpha * u[i]

# Calculate the minimum and the value of the function
f_x = 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)

# Print the results
print("1. Q-conjugate vectors:")
for i, ui in enumerate(u):
    print(f"u{i} = {ui}")

print("\n2. Conjugate Directions Method results:")
print(f"(a) Minimum x* = {x}, f(x*) = {f_x:.9f}")
print(f"(b) Number of iterations: {len(u)}")


# In[ ]:




