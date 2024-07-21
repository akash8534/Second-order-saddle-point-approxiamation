import numpy as np
from scipy.optimize import minimize

# Define the limit-state function
def G2(U):
    return ((U[0] - 3)**2 / 2**2) + ((U[1] - 2)**2 / 1**2) - 1

# Objective function to minimize (distance to the origin)
def objective(U):
    return np.linalg.norm(U)

# Constraint (the limit-state function equals zero)
def constraint(U):
    return G2(U)

# Initial guess for the MPP
U0 = np.array([0, 0])

# Define the constraint in the format required by scipy.optimize
con = {'type': 'eq', 'fun': constraint}

# Perform the optimization using SLSQP algorithm
result = minimize(objective, U0, method='SLSQP', constraints=[con], options={'disp': False})

# Display results
if result.success:
    U_mpp = result.x
    print('Most Probable Point (MPP):', U_mpp)
else:
    print('Optimization failed:', result.message)
