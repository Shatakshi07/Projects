#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import CMO_A2

# Initialize parameters and variables
x0 = np.zeros(5)
alpha = 0.00004
theta = 0.142
iterations_gradient_descent = 0
iterations_accelerated_gradient_descent = 0

# Initialize xk-1 for accelerated gradient descent
x_prev = np.zeros(5)

# Gradient Descent
x = x0
while True:
    val_at_x, grad_at_x = CMO_A2.oracle3(22888, x)
    x = x - alpha * grad_at_x
    iterations_gradient_descent += 1
    if np.linalg.norm(grad_at_x) <= 1e-4:
        break

        
# Report results
final_point_gradient_descent = x
final_function_value_gradient_descent = val_at_x


# Accelerated Gradient Descent
x = x0
while True:
    y = x + theta * (x - x_prev)
    val_at_y, grad_at_y = CMO_A2.oracle3(22888, y)
    x_prev = x.copy()
    x = y - alpha * grad_at_y
    iterations_accelerated_gradient_descent += 1
    if np.linalg.norm(grad_at_y) <= 1e-4:
        break

# Report results
final_point_accelerated_gradient_descent = x
final_function_value_accelerated_gradient_descent = val_at_y

print("Gradient Descent:")
print(f"Iterations: {iterations_gradient_descent}")
print(f"Final Point: {final_point_gradient_descent.round(9)}")
print(f"Final Function Value: {final_function_value_gradient_descent:.9f}")

print("\nAccelerated Gradient Descent:")
print(f"Iterations: {iterations_accelerated_gradient_descent}")
print(f"Final Point: {final_point_accelerated_gradient_descent.round(9)}")
print(f"Final Function Value: {final_function_value_accelerated_gradient_descent:.9f}")


# In[ ]:




