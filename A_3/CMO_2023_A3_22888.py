#------------------------QUESTION 1------------------------------------------

import subprocess
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([1,0])
m = np.matrix([[1,0],[0,1]])
tol = pow(10,-4)

 
def getgrad(x):

    str_form = [str(i) for i in x]
    outp = subprocess.run(["getGradient/getGradient", "22888", str_form[0], str_form[1]],stdout=subprocess.PIPE).stdout.decode("utf-8")
    outp = outp.split('\n')[-2]
    outp = [float(i) for i in outp.replace("[","").replace("]","").replace(" ","").strip().split(",")]
    f = outp[0]
    g = np.array([outp[1],outp[2]])
    return (f,g)

def graddescent():

    iterations = 0
    alpha = 0.2
    x= x0
    (f,g) = getgrad(x)
    gradnorm = [np.linalg.norm(g)]
    while(np.linalg.norm(g) > tol):
        x_1= np.subtract(x, alpha*g)
        x = x_1
        iterations ++
        (f,g) = getgrad(x)
        print(iterions,x,f,np.linalg.norm(g))
        gradnorm.append(np.linalg.norm(g))
    return (x,f,np.linalg.norm(g),gradnorm)

 

def qausinewtonrankone():
     f,g = get_vals(x)
    gradnorm= [np.linalg.norm(g)]
    n=0.1
    B= m
    iterations = 0
    x= x0
    while np.linalg.norm(g) > tol:
        c = -1*np.matmul(B,g)
        v = np.array([c[0,0],c[0,1]])
        alpha = backtracking(x,f,g,v,n,initial_alp = 0.9)
        x_1 = np.add(x,alpha*v)
        f1,g1 = getgradient(x_1)
        delta = np.subtract(x_1,x)
        gamma = np.subtract(g1,g)
        yi = delta - np.dot(B, gamma)
        y2 = np.outer(y1, y1)
        y3 = np.dot(y1, gamma)
        B = np.add(B,np.divide(y2, y3))
        iterations++
        f,g = f1,g1
        x = x_1
        gradnorm.append(np.linalg.norm(g))
        print(iterations, x, f, np.linalg.norm(g))

    return x, f, np.linalg.norm(g),gradnorm

 

def qausinewtonranktwo():

    n=0.1
    iterations = 0
    x = x0
    B = m
    f,g = getgradient(x)
    gradnorm = [np.linalg.norm(g)]
    while np.linalg.norm(g) > tol:
        c = -1*np.matmul(B,g)
        v= np.array([c[0,0],c[0,1]])
        alpha = backtracking(x,f,g,v,n,initial_alp = 0.9)
        x_1 = np.add(x,alpha*v)
        f1,g1 = getgradient(x_1)
        delta = np.subtract(x_1,x)
        gamma = np.subtract(g1,g)
        d1 = np.dot(delta,gamma)
        number1 = np.outer(delta,delta.T)
        d2 = np.matmul(np.matmul(gamma.T,Bk),gamma)[0,0]
        w = np.outer(gamma,gamma.T)
        w_1 = np.matmul(w,B)
        number2 = np.matmul(B,w_1)
        B = np.add(B,(1/d1)*number1)
        Bk = np.subtract(B,(1/d2)*number2)
        iterations++
        x = x_1
        f,g = f1,g1
        gradnorm.append(np.linalg.norm(g))
        print(iterations, x, f, np.linalg.norm(g))
    return x, f, np.linalg.norm(g), gradnorm

 

def backtracking(x,f,g,l,n,initial_alp):
    alpha = initial_alp
    iterations = 0
    while iterations < 100:
        s = np.add(x,alpha*l)
        (fdash,gdash) = getgradient(s)
        if fdash <f + n*alpha*np.dot(g,l):
            break
        else:
            alpha *= 0.5
        iterations++

    return alpha

 

if __name__ == "__main__":

    print("Quasi Newton Rank One")
    r1 = qausinewtonrankone()
    plt.plot(r1[-1], label='Rank 1', marker='o', linestyle='-')
    print("Quasi Newton Ran Two")
    r2 = qausinewtonranktwo()
    plt.plot(r2[-1], label='Rank 2', marker='x', linestyle='--')
    print("Gradient Descent Algorithm")
    gd = graddescent()
    plt.plot(gd[-1], label='Grad Desc.', marker='s', linestyle=':')
    plt.show()

#-------------------------QUESTION 1 END-------------------------------------------




#-------------------------QUESTION 2.4---------------------------------------------

from scipy.optimize import linprog
c = [-0.25, 1]

A = [
    [0, -1],
    [-0.5, -1],
    [1, -2]
]

b = [1, -1, -1]

res = linprog(c, A_ub=A, b_ub=b, method='highs')

if res.success:
    print("Optimal solution found:")
    print(f"x = {res.x[0]}, y = {res.x[1]}")
    print(f"Optimal value: {res.fun}")
else:
    print("Optimization failed.")


#-------------------------QUESTION 2.4 END-----------------------------------------




#-------------------------QUESTION 3-----------------------------------------------

import numpy as np
import cvxopt

data = np.loadtxt('data.txt')
labels = np.loadtxt('labels.txt')
n_samples, n_features = data.shape

K = np.dot(data, data.T)
P = cvxopt.matrix(np.outer(labels, labels) * K)
q = cvxopt.matrix(-np.ones(n_samples))
G = cvxopt.matrix(np.diag(-np.ones(n_samples)))
h = cvxopt.matrix(np.zeros(n_samples))
A = cvxopt.matrix(labels, (1, n_samples))
b = cvxopt.matrix(0.0)

solution = cvxopt.solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(solution['x'])

threshold = 1e-5 
sv_indices = np.where(alphas > threshold)[0]
support_vectors = data[sv_indices]
support_vector_labels = labels[sv_indices]
support_vector_alphas = alphas[sv_indices]
bias = np.mean(support_vector_labels - np.dot(support_vector_alphas * support_vector_labels, K[sv_indices, sv_indices]))

print("Support vectors:", len(support_vectors))
print("Bias:", bias)


#-------------------------QUESTION 3 END-------------------------------------------




#-------------------------QUESTION 4------------------------------------------------

import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2


def constraint1(x):
    return x[0] - 2 * x[1] + 2

def constraint2(x):
    return -x[0] - 2 * x[1] + 6

def constraint3(x):
    return -x[0] + 2 * x[1] + 2

def check_constraints(x):
    return all([constraint1(x) >= 0, constraint2(x) >= 0, constraint3(x) >= 0, x[0] >= 0, x[1] >= 0])


def active_set(initial_point, initial_working_set):
    max_iterations = 10
    x = initial_point
    active_set = initial_working_set
    iterations = 0

    while iterations < max_iterations:
        
        active_constraints = [{'type': 'ineq', 'fun': constraint1},
                              {'type': 'ineq', 'fun': constraint2},
                              {'type': 'ineq', 'fun': constraint3}]
        active_indices = list(active_set)
        active_constraints = [active_constraints[i] for i in active_indices if i < len(active_constraints)]
        result = minimize(objective_function, x, constraints=active_constraints)
        x = result.x
        if check_constraints(x):
            return x, active_set
        violated_constraints = [i for i, c in enumerate([constraint1(x), constraint2(x), constraint3(x)]) if c < 0]
        for idx in violated_constraints:
            if idx not in active_set:
                active_set.add(idx)

        iterations += 1

    return x, active_set

initial_point = np.array([2, 0])
initial_working_sets = [set(), {2}, {4}, {2, 4}]

for i, initial_working_set in enumerate(initial_working_sets):
    result, final_active_set = active_set(initial_point, initial_working_set)
    print(f"Iteration {i+1}: Initial Working Set: {initial_working_set}, Final Active Set: {final_active_set}, Optimal Solution: {result}")


#----------------------------QUESTION 4 END------------------------------------------------------