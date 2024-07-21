import sympy as sp

# Define the symbolic variables
U1, U2 = sp.symbols('U1 U2')

# Define the limit state function G2(U)
G2 = ((U1 - 3)**2 / 0.3**2) + ((U2 - 3)**2 / 0.2**2) - 1

# Compute the gradient (first-order partial derivatives)
grad_G2 = [sp.diff(G2, U1), sp.diff(G2, U2)]

# Compute the Hessian matrix (second-order partial derivatives)
hessian_G2 = sp.Matrix([[sp.diff(grad_G2[i], var) for var in (U1, U2)] for i in range(2)])

# Simplify the Hessian matrix
hessian_G2 = sp.simplify(hessian_G2)

# Define the Most Probable Point (MPP)
mpp = {U1: 2.75, U2: 2.88}

# Evaluate the Hessian matrix at the MPP
hessian_at_mpp = hessian_G2.subs(mpp)

# Display the Hessian matrix and its evaluated form at the MPP
print("Hessian Matrix of G2(U):")
sp.pretty_print(hessian_G2)

print("\nHessian Matrix of G2(U) evaluated at MPP (1.27, 1.49):")
sp.pretty_print(hessian_at_mpp)
