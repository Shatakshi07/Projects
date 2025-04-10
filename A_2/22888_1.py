#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import CMO_A2

def trisection_method(x1, y1, epsilon):
    t = int(np.ceil(np.log((y1 - x1) / epsilon) / np.log(3)))
    xt, yt = x1, y1
    
    for _ in range(t):
        r1 = xt + (yt - xt) / 3
        s1 = xt + 2 * (yt - xt) / 3
        
        gr1 = CMO_A2.oracle1(22888, r1)
        gs1 = CMO_A2.oracle1(22888, s1)
        
        if gr1 <= gs1:
            yt = s1
        else:
            xt = r1
    
    alpha_star = (xt + yt) / 2
    g_alpha_star = CMO_A2.oracle1(22888, alpha_star)
    
    return t, alpha_star, g_alpha_star

# Using the given initial interval [x1, y1] = [-1, 1] and tolerance epsilon = 1e-4
iterations, alpha_star, g_alpha_star = trisection_method(-1, 1, 1e-4)

# Print the results
print(f'Number of iterations: {iterations}')
print(f'Estimate of the minimizer alpha*: {alpha_star:.9f}')
print(f'Estimate of the minimum value g(alpha*): {g_alpha_star:.9f}')


# In[ ]:




