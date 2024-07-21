import numpy as np
from scipy.optimize import minimize

# Define the limit state function G2(U)
def G2(U):
    U1, U2 = U
    return ((U1 - 3)**2 / 0.3**2) + ((U2 - 3)**2 / 0.2**2) - 1

# Define the objective function (Euclidean distance from the origin)
def objective(U):
    return np.linalg.norm(U)

# Define the constraint function such that G2(U) = 0
def constraint(U):
    return G2(U)

# Initial guesses for U (try multiple guesses to avoid local minima)
initial_guesses = [
    np.array([3.0, 2.0]),
    np.array([0.0, 0.0]),
    np.array([2.0, 1.0]),
    np.array([1.5, 1.5])
]

# Define the constraint in the form required by 'minimize'
con = {'type': 'eq', 'fun': constraint}

# Container to store the best result
best_result = None

# Perform the optimization with different initial guesses
for U0 in initial_guesses:
    result = minimize(objective, U0, constraints=[con], method='SLSQP')
    if result.success:
        if best_result is None or result.fun < best_result.fun:
            best_result = result

# Print the best result
if best_result is not None:
    print("Most Probable Point (MPP):", best_result.x)
else:
    print("Optimization failed for all initial guesses")
